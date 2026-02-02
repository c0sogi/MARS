import os
import re
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    CACHE_FILE_PATH,
    IMG_SIZE,
    NUM_SLICES,
    STRIDE,
    DEPTH_MIN,
    DEPTH_MAX,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import read_dicom_robust, seed_everything

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_sorted_files(directory):
    """
    Returns a numerically sorted list of DICOM files in a directory.
    """
    if not os.path.exists(directory):
        return []
    files = [f for f in os.listdir(directory) if f.endswith(".dcm")]
    # Sort by the integer number in the filename (e.g., Image-10.dcm)
    files.sort(key=lambda x: int(re.search(r"\d+", x).group()))
    return files


def compute_anchor_index(subject_path_flair):
    """
    Implements the Raw-Integral-Greedy ROI selection strategy.
    Returns the index of the slice with the maximum intensity integral
    within the valid depth range.
    """
    full_path = os.path.join(INPUT_DIR, subject_path_flair)
    files = get_sorted_files(full_path)
    num_files = len(files)

    if num_files == 0:
        return 0

    # Define search bounds (15% - 85%)
    start_idx = int(num_files * DEPTH_MIN)
    end_idx = int(num_files * DEPTH_MAX)

    # Ensure valid range
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_files

    max_integral = -1.0
    best_idx = start_idx  # Default to start if everything fails

    # Greedy search
    for i in range(start_idx, end_idx):
        f_path = os.path.join(full_path, files[i])
        # Read image (float32)
        img = read_dicom_robust(f_path)
        # Calculate integral
        integral = np.sum(img)

        if integral > max_integral:
            max_integral = integral
            best_idx = i

    return best_idx


def generate_roi_cache(df, load_cached_data=True):
    """
    Manages the caching of ROI anchor indices.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Try to load existing cache
    if load_cached_data and os.path.exists(CACHE_FILE_PATH):
        try:
            cache_df = pd.read_parquet(CACHE_FILE_PATH)
            # Check if cache covers all current IDs
            cached_ids = set(cache_df["BraTS21ID"])
            current_ids = set(df["BraTS21ID"])

            if current_ids.issubset(cached_ids):
                return cache_df
        except Exception:
            pass  # Fallback to recomputing if cache is corrupt

    # 2. Compute missing ROIs
    print("Generating ROI cache (Raw-Integral-Greedy)...")
    roi_data = []

    # Use unique IDs to avoid redundant computation if df has duplicates
    unique_df = df.drop_duplicates(subset=["BraTS21ID"])

    for _, row in tqdm(
        unique_df.iterrows(), total=len(unique_df), desc="ROI Selection"
    ):
        anchor_idx = compute_anchor_index(row["path_FLAIR"])
        roi_data.append({"BraTS21ID": row["BraTS21ID"], "anchor_idx": anchor_idx})

    cache_df = pd.DataFrame(roi_data)

    # 3. Save to parquet
    cache_df.to_parquet(CACHE_FILE_PATH, index=False)

    return cache_df


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class BraTSDataset(Dataset):
    def __init__(self, dataframe, phase="train", load_cached_roi=True):
        """
        Args:
            dataframe (pd.DataFrame): Metadata dataframe.
            phase (str): 'train', 'val', or 'test'.
            load_cached_roi (bool): Whether to use cached ROI indices.
        """
        self.phase = phase
        self.df = dataframe.reset_index(drop=True)

        # Merge ROI cache
        cache_df = generate_roi_cache(self.df, load_cached_data=load_cached_roi)
        self.df = self.df.merge(cache_df, on="BraTS21ID", how="left")

        # Fill missing anchors (should not happen if cache gen works, but for safety)
        self.df["anchor_idx"] = self.df["anchor_idx"].fillna(0).astype(int)

        # Define Augmentations
        if self.phase == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # Rotate +/- 15 degrees, keep background 0
                    A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose([ToTensorV2()])

        # Modality order matches model expectation: FLAIR, T1w, T1wCE, T2w
        self.modalities = ["path_FLAIR", "path_T1w", "path_T1wCE", "path_T2w"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        brats_id = row["BraTS21ID"]
        anchor_idx = row["anchor_idx"]

        # Determine slice indices: [Anchor-5, Anchor, Anchor+5]
        # We need to clamp these later based on actual file count per modality
        relative_indices = [-STRIDE, 0, STRIDE]

        channels = []

        for mod_col in self.modalities:
            dir_path = os.path.join(INPUT_DIR, row[mod_col])
            files = get_sorted_files(dir_path)
            num_files = len(files)

            if num_files == 0:
                # Handle missing modality: return zeros
                for _ in range(NUM_SLICES):
                    channels.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
                continue

            # Load the 3 slices
            for rel_idx in relative_indices:
                target_idx = anchor_idx + rel_idx
                # Clamp index
                target_idx = max(0, min(target_idx, num_files - 1))

                file_path = os.path.join(dir_path, files[target_idx])
                img = read_dicom_robust(file_path)

                # Conservative Min-Max Scaling [0, 1]
                # Avoid division by zero
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
                else:
                    img = np.zeros_like(img)  # Flat image becomes 0

                channels.append(img)

        # Stack channels: (H, W, 12)
        # Albumentations expects HWC
        volume = np.stack(channels, axis=-1)

        # Apply Augmentations
        augmented = self.transform(image=volume)
        volume_tensor = augmented["image"]  # (12, H, W) due to ToTensorV2

        # Get Label
        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
        else:
            target = torch.tensor(-1.0, dtype=torch.float32)  # Dummy for test

        return volume_tensor, target, brats_id


# -----------------------------------------------------------------------------
# Data Loader Factory
# -----------------------------------------------------------------------------


def get_dataloader(
    dataframe, phase, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True
):
    """
    Creates a DataLoader for the given phase.
    """
    dataset = BraTSDataset(dataframe, phase=phase)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(phase == "train"),  # Drop last incomplete batch only for training
    )

    return loader
