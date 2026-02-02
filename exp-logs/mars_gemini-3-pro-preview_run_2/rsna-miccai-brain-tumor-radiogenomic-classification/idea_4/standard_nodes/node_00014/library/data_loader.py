import os
import re
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import rasterio
from library.config import Config
from library.utils import seed_everything

# Suppress rasterio NotGeoreferencedWarning as we are reading medical images
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def natural_sort_key(s):
    """
    Sorts strings containing numbers naturally (e.g. Image-2 before Image-10).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def load_dicom_slice(path):
    """
    Reads a DICOM file from disk. Tries rasterio first, falls back to cv2.
    Returns a numpy array or a zero array if reading fails.
    """
    try:
        # Try rasterio (GDAL based)
        with rasterio.open(path) as src:
            img = src.read(1)
            return img
    except Exception:
        pass

    try:
        # Fallback to OpenCV
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    # Return 256x256 zeros if everything fails (fallback)
    return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def get_sorted_files(dir_path):
    """Returns sorted list of file paths in a directory."""
    if not os.path.exists(dir_path):
        return []
    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
    files.sort(key=natural_sort_key)
    return files


def preprocess_metadata(
    df, cache_name="processed_metadata.parquet", load_cached_data=True, debug=False
):
    """
    Calculates the 'Peak Intensity Index' for FLAIR modality for each subject.
    Caches the result to disk to avoid re-scanning files.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # 1. Load Cache if requested and available
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached metadata from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute from scratch
    # print("Computing peak intensity indices (this may take a while)...")

    peak_indices = []
    flair_counts = []

    # If debug, only process a subset
    iterator_df = df if not debug else df.head(Config.DEBUG_SAMPLE_SIZE)

    for idx, row in iterator_df.iterrows():
        flair_dir = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        files = get_sorted_files(flair_dir)
        count = len(files)

        if count == 0:
            peak_indices.append(0)
            flair_counts.append(0)
            continue

        # Read all slices to find max intensity
        # Optimization: Read middle 60% to avoid noise at ends and speed up
        start_idx = int(count * 0.2)
        end_idx = int(count * 0.8)
        if end_idx <= start_idx:
            start_idx, end_idx = 0, count

        max_intensity = -1
        best_idx = count // 2  # Default to middle

        for i in range(start_idx, end_idx):
            f_path = os.path.join(flair_dir, files[i])
            img = load_dicom_slice(f_path)
            # Use mean intensity of non-zero pixels or just sum
            intensity = np.sum(img)
            if intensity > max_intensity:
                max_intensity = intensity
                best_idx = i

        peak_indices.append(best_idx)
        flair_counts.append(count)

    # Update DataFrame
    # Note: If debug was used, we only have data for the subset.
    # For simplicity in this implementation, we assume full run or handle subset carefully.
    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()

    df["flair_peak_index"] = peak_indices
    df["flair_slice_count"] = flair_counts

    # 3. Save to Cache
    if not debug:  # Don't overwrite cache with debug data
        df.to_parquet(cache_path, index=False)

    return df


class BraTSDataset(Dataset):
    def __init__(self, df, phase="train", transform=None):
        """
        Args:
            df: DataFrame containing metadata and 'flair_peak_index'.
            phase: 'train', 'val', or 'test'.
            transform: Optional albumentations transforms.
        """
        self.df = df
        self.phase = phase
        self.transform = transform
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.df)

    def _load_volume_slice(self, row, mod, relative_idx, flair_count):
        """
        Loads a specific slice from a modality, handling depth alignment.
        """
        path_col = f"path_{mod}"
        dir_path = os.path.join(Config.INPUT_DIR, row[path_col])
        files = get_sorted_files(dir_path)

        if not files:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Map FLAIR index to current modality index
        mod_count = len(files)
        if flair_count > 0:
            target_idx = int(relative_idx * (mod_count / flair_count))
        else:
            target_idx = 0

        # Clamp
        target_idx = max(0, min(target_idx, mod_count - 1))

        # Load
        img_path = os.path.join(dir_path, files[target_idx])
        img = load_dicom_slice(img_path)

        # Resize
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

        # Normalize (Min-Max)
        if img.max() > 0:
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = img  # Keep zeros

        return img.astype(np.float32)

    def _get_indices(self, center, stride):
        """Generates 3 indices [center-stride, center, center+stride]"""
        return [center - stride, center, center + stride]

    def _construct_tensor(self, row, center_idx, stride):
        """
        Constructs a (12, H, W) tensor: 4 modalities x 3 slices.
        """
        indices = self._get_indices(center_idx, stride)
        flair_count = row["flair_slice_count"]

        channels = []
        for mod in self.modalities:
            for idx in indices:
                # Handle boundary conditions for indices
                safe_idx = max(0, min(idx, flair_count - 1))
                img = self._load_volume_slice(row, mod, safe_idx, flair_count)
                channels.append(img)

        # Stack -> (12, H, W)
        tensor = np.stack(channels, axis=0)
        return torch.tensor(tensor, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        peak_idx = row["flair_peak_index"]

        # --- TEST PHASE (TTA) ---
        if self.phase == "test":
            # Generate 3 views with different strides for TTA
            # Strides: 4, 5, 6
            views = []
            for s in [4, 5, 6]:
                view = self._construct_tensor(row, peak_idx, s)
                views.append(view)

            # Stack views -> (3, 12, H, W)
            tta_tensor = torch.stack(views, dim=0)
            return tta_tensor, row["BraTS21ID"]

        # --- TRAIN/VAL PHASE ---
        label = row["MGMT_value"]

        # View A: Fixed Stride, Centered at Peak
        view_a = self._construct_tensor(row, peak_idx, Config.BASE_STRIDE)

        # Standard Supervised Return (Cite Lesson 00012: Avoid Consistency Regularization on sparse inputs)
        return view_a, torch.tensor(label, dtype=torch.float32)


def get_dataloaders(train_df, val_df, test_df, debug=False):
    """
    Factory function to create dataloaders.
    Handles caching of peak intensity metadata.
    """
    # 1. Preprocess Metadata (Peak Detection & Caching)
    # We process train and val together or separately.
    # For simplicity and to use the cache effectively, we process them.

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Process Train
    train_df = preprocess_metadata(
        train_df,
        cache_name="train_processed.parquet",
        load_cached_data=True,
        debug=debug,
    )

    # Process Val
    val_df = preprocess_metadata(
        val_df, cache_name="val_processed.parquet", load_cached_data=True, debug=debug
    )

    # Process Test
    test_df = preprocess_metadata(
        test_df, cache_name="test_processed.parquet", load_cached_data=True, debug=debug
    )

    # 2. Create Datasets
    train_ds = BraTSDataset(train_df, phase="train")
    val_ds = BraTSDataset(val_df, phase="val")
    test_ds = BraTSDataset(test_df, phase="test")

    # 3. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
