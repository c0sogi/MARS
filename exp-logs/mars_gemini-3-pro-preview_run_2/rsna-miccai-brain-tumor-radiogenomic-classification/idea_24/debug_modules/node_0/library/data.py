import os
import re
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    IMG_SIZE,
    NUM_SLICES,
    STRIDE,
    ROI_DEPTH_MIN,
    ROI_DEPTH_MAX,
    ROTATION_DEGREES,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import seed_everything

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., Image-1.dcm, Image-2.dcm, Image-10.dcm).
    """
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def read_dicom_raw(path, img_size=IMG_SIZE):
    """
    Reads DICOM pixel data using a raw binary tail-read to bypass headers.
    Infers resolution (512x512 vs 256x256) based on file size.
    Resizes to img_size using Area Interpolation.
    Returns float32 numpy array.
    """
    try:
        file_size = os.path.getsize(path)

        # Heuristic to determine dimensions based on file size
        # 512*512*2 bytes = 524,288 bytes. Files ~525kB are 512x512.
        # 256*256*2 bytes = 131,072 bytes. Files ~132kB are 256x256.
        if file_size > 200000:
            shape = (512, 512)
        else:
            shape = (256, 256)

        num_pixels = shape[0] * shape[1]
        num_bytes = num_pixels * 2  # uint16 = 2 bytes

        # Raw binary read from the end of the file
        with open(path, "rb") as f:
            f.seek(-num_bytes, 2)
            data = f.read()

        if len(data) < num_bytes:
            # Fallback for corrupted/unexpected files
            return np.zeros((img_size, img_size), dtype=np.float32)

        # Convert bytes to numpy array
        img = np.frombuffer(data, dtype=np.uint16).reshape(shape)
        img = img.astype(np.float32)

        # Resize if necessary using Area Interpolation (best for downsampling)
        if img.shape[0] != img_size:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        return img

    except Exception:
        # Return black image on failure
        return np.zeros((img_size, img_size), dtype=np.float32)


def get_roi_cache(df, load_cached_data=True):
    """
    Manages the caching of the 'Anchor Slice' index for each subject.
    The anchor is the slice with the maximum intensity integral in the 15-85% depth range of FLAIR.

    Args:
        df (pd.DataFrame): Metadata dataframe containing BraTS21ID and path_FLAIR.
        load_cached_data (bool): Whether to attempt loading from disk.

    Returns:
        dict: Mapping {BraTS21ID: anchor_index}
    """
    cache_path = os.path.join(WORKING_DIR, "roi_cache.parquet")
    cache = {}

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            cache = cache_df.set_index("BraTS21ID")["anchor_index"].to_dict()
        except Exception:
            cache = {}

    # 2. Identify missing IDs
    unique_ids = df["BraTS21ID"].unique()
    missing_ids = [uid for uid in unique_ids if uid not in cache]

    if not missing_ids:
        return cache

    # 3. Compute anchors for missing IDs
    print(f"Computing ROI anchors for {len(missing_ids)} subjects...")

    new_entries = {}
    for uid in missing_ids:
        try:
            row = df[df["BraTS21ID"] == uid].iloc[0]
            flair_path_rel = row["path_FLAIR"]
            flair_dir = os.path.join(INPUT_DIR, flair_path_rel)

            # List and sort files
            if os.path.exists(flair_dir):
                files = os.listdir(flair_dir)
                files.sort(key=natural_sort_key)
            else:
                files = []

            num_files = len(files)

            if num_files == 0:
                new_entries[uid] = 0
                continue

            # Define search range (15% to 85%)
            start_idx = int(num_files * ROI_DEPTH_MIN)
            end_idx = int(num_files * ROI_DEPTH_MAX)

            if start_idx >= end_idx:
                start_idx = 0
                end_idx = num_files

            max_integral = -1.0
            best_idx = start_idx

            # Scan the range
            for i in range(start_idx, end_idx):
                f_path = os.path.join(flair_dir, files[i])
                # Read image (resized is fine for integral calculation)
                img = read_dicom_raw(f_path, img_size=IMG_SIZE)
                current_integral = np.sum(img)

                if current_integral > max_integral:
                    max_integral = current_integral
                    best_idx = i

            new_entries[uid] = best_idx

        except Exception:
            new_entries[uid] = 0

    # 4. Update and Save Cache
    cache.update(new_entries)

    # Save as Parquet
    full_cache_df = pd.DataFrame(
        list(cache.items()), columns=["BraTS21ID", "anchor_index"]
    )
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    full_cache_df.to_parquet(cache_path, index=False)

    return cache


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the given phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(
                    limit=ROTATION_DEGREES,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------


class BraTSDataset(Dataset):
    def __init__(self, df, roi_cache, phase="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            roi_cache (dict): Dictionary mapping BraTS21ID to anchor slice index.
            phase (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transforms to apply.
        """
        self.df = df.reset_index(drop=True)
        self.roi_cache = roi_cache
        self.phase = phase
        self.transform = transform
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # 1. Get Anchor Index
        anchor_idx = self.roi_cache.get(subject_id, 0)

        all_channels = []

        # 2. Iterate Modalities
        for mod in self.modalities:
            dir_rel_path = row[f"path_{mod}"]
            mod_dir = os.path.join(INPUT_DIR, dir_rel_path)

            # Get sorted file list
            # Note: We assume file existence based on metadata checks.
            try:
                files = os.listdir(mod_dir)
                files.sort(key=natural_sort_key)
            except FileNotFoundError:
                files = []

            num_files = len(files)

            # 3. Extract Slices using Stride
            for s in STRIDE:
                target_idx = anchor_idx + s

                # Edge Clamping
                if target_idx < 0:
                    target_idx = 0
                elif target_idx >= num_files:
                    target_idx = max(0, num_files - 1)

                if num_files > 0:
                    img_path = os.path.join(mod_dir, files[target_idx])
                    img = read_dicom_raw(img_path, img_size=IMG_SIZE)
                else:
                    # Handle empty directory case
                    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

                # 4. Independent Per-Slice Min-Max Normalization
                min_val = img.min()
                max_val = img.max()

                if max_val > min_val:
                    img = (img - min_val) / (max_val - min_val)
                else:
                    img = np.zeros_like(img)

                all_channels.append(img)

        # Stack channels: (20, H, W)
        # 4 modalities * 5 slices = 20 channels
        img_tensor = np.stack(all_channels, axis=0)

        # 5. Augmentation
        # Albumentations expects (H, W, C)
        img_tensor = np.transpose(img_tensor, (1, 2, 0))

        if self.transform:
            augmented = self.transform(image=img_tensor)
            img_tensor = augmented["image"]  # Returns (C, H, W) tensor via ToTensorV2
        else:
            # Fallback if no transform provided
            img_tensor = torch.from_numpy(np.transpose(img_tensor, (2, 0, 1)))

        # 6. Get Label
        if "MGMT_value" in row:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
        else:
            # Test set
            label = torch.tensor(-1.0, dtype=torch.float32)

        return img_tensor, label


# ------------------------------------------------------------------------------
# Data Loader Factory
# ------------------------------------------------------------------------------


def get_dataloaders(train_df=None, val_df=None, test_df=None, load_cached_data=True):
    """
    Constructs DataLoaders for train, val, and test sets.
    Ensures ROI cache is populated for all provided dataframes.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        test_df (pd.DataFrame): Test metadata.
        load_cached_data (bool): Whether to use cached ROI indices.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Consolidate DFs to update cache efficiently
    dfs = []
    if train_df is not None:
        dfs.append(train_df)
    if val_df is not None:
        dfs.append(val_df)
    if test_df is not None:
        dfs.append(test_df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        # Drop duplicates based on ID to avoid redundant processing
        combined_df = combined_df.drop_duplicates(subset=["BraTS21ID"])
        roi_cache = get_roi_cache(combined_df, load_cached_data=load_cached_data)
    else:
        roi_cache = {}

    loaders = []

    # 2. Create Train Loader
    if train_df is not None:
        train_ds = BraTSDataset(
            train_df, roi_cache, phase="train", transform=get_transforms("train")
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        loaders.append(train_loader)
    else:
        loaders.append(None)

    # 3. Create Val Loader
    if val_df is not None:
        val_ds = BraTSDataset(
            val_df, roi_cache, phase="val", transform=get_transforms("val")
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        loaders.append(val_loader)
    else:
        loaders.append(None)

    # 4. Create Test Loader
    if test_df is not None:
        test_ds = BraTSDataset(
            test_df, roi_cache, phase="test", transform=get_transforms("test")
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        loaders.append(test_loader)
    else:
        loaders.append(None)

    return tuple(loaders)
