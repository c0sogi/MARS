import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
import albumentations as A
from library.config import Config

# -----------------------------------------------------------------------------
# Caching Utilities
# -----------------------------------------------------------------------------


def get_dicom_paths(metadata_df, root_dir, cache_name, load_cached_data=True):
    """
    Retrieves and caches the sorted list of DICOM file paths for each study.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'StudyInstanceUID' and 'image_path'.
        root_dir (str): Root directory for images (e.g., ./input).
        cache_name (str): Identifier for the cache file (e.g., 'train_paths').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Mapping of StudyInstanceUID -> List[Absolute File Paths]
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}_cache.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            # Convert back to dict
            path_dict = cached_df.set_index("StudyInstanceUID")["file_paths"].to_dict()
            # Ensure paths are lists (parquet might store as array)
            path_dict = {k: list(v) for k, v in path_dict.items()}
            return path_dict
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    path_dict = {}
    study_uids = metadata_df["StudyInstanceUID"].unique()

    for uid in study_uids:
        # Construct path to study directory
        # metadata 'image_path' is relative to input root, e.g., "train_images/UID"
        # We need to find the row for this UID to get the image_path
        # Assuming metadata_df is unique on StudyInstanceUID or we take the first
        rel_path = metadata_df[metadata_df["StudyInstanceUID"] == uid].iloc[0][
            "image_path"
        ]
        study_dir = os.path.join(root_dir, rel_path)

        if not os.path.exists(study_dir):
            path_dict[uid] = []
            continue

        # Get all .dcm files
        # We use os.listdir for speed, then sort
        try:
            files = [f for f in os.listdir(study_dir) if f.endswith(".dcm")]
            # Sort by instance number (filename usually is '1.dcm', '10.dcm')
            # We assume filename is integer-like
            files.sort(key=lambda x: int(os.path.splitext(x)[0]))

            full_paths = [os.path.join(study_dir, f) for f in files]
            path_dict[uid] = full_paths
        except Exception:
            path_dict[uid] = []

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    # Convert dict to DF for parquet
    cache_df = pd.DataFrame(
        [{"StudyInstanceUID": k, "file_paths": v} for k, v in path_dict.items()]
    )
    cache_df.to_parquet(cache_path, index=False)

    return path_dict


# -----------------------------------------------------------------------------
# Image Processing Utilities
# -----------------------------------------------------------------------------


def load_dicom_slice(path, image_size):
    """
    Reads a DICOM file, applies windowing/normalization, and resizes.
    Returns a 2D numpy array (H, W) in range [0, 1].
    """
    try:
        dcm = pydicom.dcmread(path)
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Apply Rescale Slope/Intercept if present to get HU
        slope = getattr(dcm, "RescaleSlope", 1.0)
        intercept = getattr(dcm, "RescaleIntercept", 0.0)
        pixel_array = pixel_array * slope + intercept

        # Bone Windowing
        # Center (WL): 500, Width (WW): 2000 => Range [-500, 1500]
        # This is a standard approximation for spine bone visualization
        center = 500
        width = 2000
        low = center - width / 2
        high = center + width / 2

        img = np.clip(pixel_array, low, high)
        img = (img - low) / (high - low)  # Normalize to [0, 1]

    except Exception:
        # Fallback for corrupt files or missing headers
        img = np.zeros(image_size, dtype=np.float32)

    # Resize
    if img.shape != image_size:
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)

    return img


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class RSNADataset(Dataset):
    def __init__(self, metadata_df, path_dict, transform=None, is_train=False):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata with targets and UIDs.
            path_dict (dict): Pre-computed map of UID -> list of file paths.
            transform (albumentations.Compose): Augmentation pipeline.
            is_train (bool): Whether to load targets.
        """
        self.metadata = metadata_df
        self.path_dict = path_dict
        self.transform = transform
        self.is_train = is_train

        # Target columns in order: C1..C7, Patient_Overall
        self.target_cols = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

        # Pre-filter metadata to only include studies with paths
        valid_uids = set(path_dict.keys())
        self.metadata = self.metadata[
            self.metadata["StudyInstanceUID"].isin(valid_uids)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uid = row["StudyInstanceUID"]
        all_paths = self.path_dict.get(uid, [])

        # --- 1. Sequence Sampling ---
        num_slices = len(all_paths)
        seq_len = Config.SEQ_LEN

        if num_slices == 0:
            # Handle edge case of empty directory
            indices = np.zeros(seq_len, dtype=int)
            # Create dummy paths? No, handle in loading loop
        elif num_slices < seq_len:
            # Pad if fewer slices than sequence length
            # We use linear interpolation of indices to stretch
            indices = np.linspace(0, num_slices - 1, seq_len).round().astype(int)
        else:
            # Uniformly sample if more slices
            indices = np.linspace(0, num_slices - 1, seq_len).round().astype(int)

        # --- 2. 2.5D Stacking & Loading ---
        # We need to load 3 slices for each sampled index: z-1, z, z+1
        # To optimize I/O, we identify all unique physical files needed first

        unique_indices = set()
        for i in indices:
            unique_indices.add(max(0, i - 1))
            unique_indices.add(i)
            unique_indices.add(min(num_slices - 1, i + 1))

        # Load unique slices into memory
        loaded_slices = {}
        for i in unique_indices:
            if num_slices == 0:
                loaded_slices[i] = np.zeros(Config.IMAGE_SIZE, dtype=np.float32)
            else:
                loaded_slices[i] = load_dicom_slice(all_paths[i], Config.IMAGE_SIZE)

        # --- 3. Construct Sequence & Augment ---
        # We must apply the SAME geometric augmentation to all slices in the sequence
        # to preserve anatomical alignment.

        replay_data = None
        sequence_tensor = np.zeros(
            (seq_len, Config.IN_CHANNELS, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
            dtype=np.float32,
        )

        for t, center_idx in enumerate(indices):
            # Form 2.5D stack
            # Channel 0: z-1, Channel 1: z, Channel 2: z+1
            idx_prev = max(0, center_idx - 1)
            idx_next = min(num_slices - 1, center_idx + 1)

            img_stack = np.stack(
                [
                    loaded_slices[idx_prev],
                    loaded_slices[center_idx],
                    loaded_slices[idx_next],
                ],
                axis=-1,
            )  # (H, W, 3)

            # Apply Augmentation
            if self.transform:
                if replay_data is None:
                    # First slice: generate parameters and save replay
                    res = self.transform(image=img_stack)
                    img_aug = res["image"]
                    if isinstance(self.transform, A.ReplayCompose):
                        replay_data = res["replay"]
                else:
                    # Subsequent slices: replay parameters
                    res = A.ReplayCompose.replay(replay_data, image=img_stack)
                    img_aug = res["image"]
            else:
                img_aug = img_stack

            # Transpose to (C, H, W) for PyTorch
            # Albumentations outputs (H, W, C)
            sequence_tensor[t] = img_aug.transpose(2, 0, 1)

        # Convert to Tensor
        sequence_tensor = torch.tensor(sequence_tensor, dtype=torch.float32)

        # --- 4. Targets ---
        if self.is_train:
            labels = row[self.target_cols].values.astype(np.float32)
            return sequence_tensor, torch.tensor(labels)
        else:
            # For test, we might need the UID to map predictions back
            return sequence_tensor, uid


# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------


def get_loaders(load_cached_data=True, debug=Config.DEBUG, debug_size=None):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached file paths.
        debug (bool): If True, subsets data for quick testing.
        debug_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if debug_size is None:
        debug_size = Config.DEBUG_DATA_SIZE

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]
        print(
            f"DEBUG Mode: Using {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples."
        )

    # 2. Prepare File Paths (Cached)
    # We combine all UIDs to scan directories efficiently or scan per split
    # Scanning all at once is cleaner if they share root, but they have different roots in Config
    train_paths = get_dicom_paths(
        pd.concat([train_df, val_df]), Config.ROOT_DIR, "train_paths", load_cached_data
    )
    test_paths = get_dicom_paths(
        test_df, Config.ROOT_DIR, "test_paths", load_cached_data
    )

    # 3. Define Augmentations
    # We use ReplayCompose to ensure the same random transform is applied to every slice in the stack
    train_transform = A.ReplayCompose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            # CoarseDropout or similar could be added, but keeping it simple for stability
        ]
    )

    # No augmentation for validation/test, just resizing (handled in loading)
    val_transform = None

    # 4. Create Datasets
    train_dataset = RSNADataset(
        train_df, train_paths, transform=train_transform, is_train=True
    )

    val_dataset = RSNADataset(
        val_df, train_paths, transform=val_transform, is_train=True
    )

    test_dataset = RSNADataset(
        test_df, test_paths, transform=val_transform, is_train=False
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
