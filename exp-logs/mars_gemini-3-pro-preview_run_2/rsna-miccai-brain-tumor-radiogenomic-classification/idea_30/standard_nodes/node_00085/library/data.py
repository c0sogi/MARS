import os
import re
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# Import from the provided library
from library.utils import (
    read_dicom_image,
    resize_image,
    normalize_image,
    save_cache,
    load_cache,
)

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_opt"
ROI_CACHE_FILE = os.path.join(CACHE_DIR, "roi_cache.parquet")
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def extract_id(filename):
    """Extracts the integer ID from a filename like 'Image-123.dcm'."""
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1


def get_image_ids(directory):
    """Returns a sorted list of integer Image IDs found in the directory."""
    if not os.path.exists(directory):
        return []
    files = os.listdir(directory)
    ids = [extract_id(f) for f in files if f.endswith(".dcm")]
    return sorted([i for i in ids if i != -1])


# -----------------------------------------------------------------------------
# ROI Calculation Logic
# -----------------------------------------------------------------------------
def compute_flair_integral_roi(metadata_dfs, load_cached_data=True):
    """
    Computes the anchor slice index using FLAIR Integral Consensus.
    Cite Lesson 00053: Derive slice selection from a single dominant reference modality.
    Cite Lesson 00038: Use integral statistics (Sum) over extremal ones.

    Args:
        metadata_dfs (list): List of pandas DataFrames (train, val, test).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'BraTS21ID', 'anchor_id', 'sorted_ids'.
    """
    # 1. Try Load Cache
    if load_cached_data:
        cached_df = load_cache(ROI_CACHE_FILE)
        if cached_df is not None:
            print(f"Loaded ROI cache from {ROI_CACHE_FILE}")
            return cached_df

    print("Computing FLAIR-Integral ROI cache (this may take a few minutes)...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 2. Consolidate Subject List
    all_subjects = []
    seen_ids = set()

    for df in metadata_dfs:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            bid = row["BraTS21ID"]
            if bid not in seen_ids:
                seen_ids.add(bid)
                all_subjects.append(row)

    results = []

    # 3. Process Each Subject
    for row in all_subjects:
        bid = row["BraTS21ID"]

        # Define path for ROI modality (FLAIR Only)
        path_flair = os.path.join(INPUT_DIR, row["path_FLAIR"])

        # Get all available IDs
        ids_flair = get_image_ids(path_flair)

        if not ids_flair:
            # Fallback for empty folders
            results.append({"BraTS21ID": bid, "anchor_id": 1, "sorted_ids": [1]})
            continue

        # Compute Integral Profiles for FLAIR
        profile_flair = []

        for i in ids_flair:
            p = os.path.join(path_flair, f"Image-{i}.dcm")
            img = read_dicom_image(p)
            val_f = np.sum(img)
            profile_flair.append(val_f)

        # Use raw sum profile (no normalization needed for single modality)
        consensus = np.array(profile_flair)

        # Restrict to 15% - 85% depth (Cite Lesson 00080)
        n = len(ids_flair)
        start = int(n * 0.15)
        end = int(n * 0.85)

        # Handle edge case where range is invalid
        if start >= end:
            start = 0
            end = n

        subset = consensus[start:end]

        if len(subset) > 0:
            argmax_rel = np.argmax(subset)
            argmax_abs = start + argmax_rel
            anchor_id = ids_flair[argmax_abs]
        else:
            # Fallback to middle
            anchor_id = ids_flair[n // 2]

        results.append(
            {"BraTS21ID": bid, "anchor_id": anchor_id, "sorted_ids": ids_flair}
        )

    # 4. Save Cache
    df_cache = pd.DataFrame(results)
    save_cache(df_cache, ROI_CACHE_FILE)
    print("ROI cache computation complete.")
    return df_cache


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class MGMTDataset(Dataset):
    def __init__(self, df, roi_cache_df, transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            roi_cache_df (pd.DataFrame): Dataframe with ROI info.
            transform (albumentations.Compose): Augmentations.
        """
        self.df = df
        self.transform = transform

        # Convert cache to dictionary for O(1) access
        self.cache = {}
        if roi_cache_df is not None and not roi_cache_df.empty:
            for _, row in roi_cache_df.iterrows():
                s_ids = row["sorted_ids"]
                # Ensure list type
                if isinstance(s_ids, np.ndarray):
                    s_ids = s_ids.tolist()
                self.cache[row["BraTS21ID"]] = {
                    "anchor_id": row["anchor_id"],
                    "sorted_ids": s_ids,
                }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bid = row["BraTS21ID"]

        # 1. Determine Target Slice IDs
        if bid in self.cache:
            entry = self.cache[bid]
            anchor_id = entry["anchor_id"]
            sorted_ids = entry["sorted_ids"]
        else:
            # Fallback
            anchor_id = 1
            sorted_ids = [1]

        if not sorted_ids:
            sorted_ids = [1]
            anchor_id = 1

        # Find index of anchor
        try:
            anchor_idx = sorted_ids.index(anchor_id)
        except ValueError:
            anchor_idx = len(sorted_ids) // 2

        # Stride 5 with Edge Clamping
        indices = [anchor_idx - 5, anchor_idx, anchor_idx + 5]
        clamped_indices = [max(0, min(len(sorted_ids) - 1, i)) for i in indices]
        target_ids = [sorted_ids[i] for i in clamped_indices]

        # 2. Load Images (12 Channels)
        # Order: [Mod1_S1, Mod1_S2, Mod1_S3, Mod2_S1, ...]
        # Modalities: FLAIR, T1w, T1wCE, T2w

        channels = []
        for mod in MODALITIES:
            # Path from metadata
            rel_path = row[f"path_{mod}"]
            full_dir = os.path.join(INPUT_DIR, rel_path)

            for tid in target_ids:
                # Construct filename
                fpath = os.path.join(full_dir, f"Image-{tid}.dcm")

                # Read -> Resize -> Normalize
                img = read_dicom_image(fpath)
                img = resize_image(img, (224, 224))
                img = normalize_image(img)

                channels.append(img)

        # Stack to (H, W, C) -> (224, 224, 12)
        image = np.stack(channels, axis=-1)

        # 3. Augmentations
        if self.transform:
            # Albumentations works on (H, W, C)
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 4. Convert to Tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # 5. Label
        if "MGMT_value" in row:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
        else:
            label = torch.tensor(-1.0, dtype=torch.float32)

        return image, label


# -----------------------------------------------------------------------------
# Data Loading Function
# -----------------------------------------------------------------------------
def get_dataloaders(
    train_metadata_path="./metadata/train.csv",
    val_metadata_path="./metadata/val.csv",
    test_metadata_path="./metadata/test.csv",
    batch_size=32,
    num_workers=4,
    load_cached_roi=True,
):
    """
    Constructs DataLoaders for train, val, and test sets.
    """
    # Load Metadata
    df_train = pd.read_csv(train_metadata_path)
    df_val = pd.read_csv(val_metadata_path)
    df_test = pd.read_csv(test_metadata_path)

    # Compute/Load ROIs
    roi_cache = compute_flair_integral_roi(
        [df_train, df_val, df_test], load_cached_data=load_cached_roi
    )

    # Define Transforms
    # Training: HFlip, VFlip, Rotate +/- 15 deg with Reflection Padding
    # Cite Lesson 00048: Reflection padding prevents stagnation compared to zero padding
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
        ]
    )

    # Validation/Test: No geometric transforms (TTA handled in inference loop usually)
    val_transform = None

    # Instantiate Datasets
    train_dataset = MGMTDataset(df_train, roi_cache, transform=train_transform)
    val_dataset = MGMTDataset(df_val, roi_cache, transform=val_transform)
    test_dataset = MGMTDataset(df_test, roi_cache, transform=val_transform)

    # Instantiate Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
