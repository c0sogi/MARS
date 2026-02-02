import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library import config


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays, and caches it.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: Dictionary containing 'ids', 'images', 'angles', 'labels'.
    """
    cache_path = config.CACHE_PATH

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            return {
                "ids": cached["ids"],
                "images": cached["images"],
                "angles": cached["angles"],
                "labels": cached["labels"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print("Processing raw data from JSON files...")

    # Load Train
    with open(config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)

    # Load Test
    with open(config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # Combine for unified processing
    # We mark test labels as -1
    all_data = train_data + test_data

    # Pre-allocate arrays
    n_samples = len(all_data)
    ids = []
    images = np.zeros((n_samples, 75, 75, 2), dtype=np.float32)
    angles = np.full(n_samples, np.nan, dtype=np.float32)
    labels = np.full(n_samples, -1, dtype=np.float32)

    for i, entry in enumerate(all_data):
        ids.append(entry["id"])

        # Process Bands
        # Raw data is list of 5625 floats
        b1 = np.array(entry["band_1"]).reshape(75, 75)
        b2 = np.array(entry["band_2"]).reshape(75, 75)
        images[i, :, :, 0] = b1
        images[i, :, :, 1] = b2

        # Process Angle
        # "na" is converted to NaN
        inc_angle = entry["inc_angle"]
        if inc_angle != "na":
            angles[i] = float(inc_angle)

        # Process Label (only exists in train)
        if "is_iceberg" in entry:
            labels[i] = entry["is_iceberg"]

    ids = np.array(ids)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, ids=ids, images=images, angles=angles, labels=labels)
    print(f"Data processed and cached to {cache_path}")

    return {"ids": ids, "images": images, "angles": angles, "labels": labels}


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, stats=None, transform=False):
        """
        Args:
            images (np.ndarray): (N, 75, 75, 2) array of Band 1 and Band 2.
            angles (np.ndarray): (N,) array of incidence angles.
            labels (np.ndarray, optional): (N,) array of targets.
            stats (dict): Dictionary containing 'min' and 'max' arrays for normalization.
            transform (bool): Whether to apply augmentations.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.stats = stats
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve Data
        # Shape: (75, 75, 2)
        img_raw = self.images[idx]
        angle = self.angles[idx]

        # 2. Construct 3rd Channel (Mean)
        # Shape: (75, 75)
        b1 = img_raw[:, :, 0]
        b2 = img_raw[:, :, 1]
        b3 = (b1 + b2) / 2.0

        # Stack to (75, 75, 3)
        img_stack = np.dstack((b1, b2, b3))

        # 3. Normalization (Independent Per-Channel Min-Max)
        if self.stats:
            min_vals = self.stats["min"]  # Shape (3,)
            max_vals = self.stats["max"]  # Shape (3,)

            # Broadcast subtraction/division
            img_stack = (img_stack - min_vals) / (max_vals - min_vals + 1e-8)

            # Clip to [0, 1] to handle potential outliers in test set
            img_stack = np.clip(img_stack, 0.0, 1.0)

        # 4. Convert to Tensor (C, H, W)
        img_tensor = torch.from_numpy(img_stack).float().permute(2, 0, 1)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # 5. Augmentation
        if self.transform:
            # Rotational Invariance: 0, 90, 180, 270
            k = torch.randint(0, 4, (1,)).item()
            img_tensor = torch.rot90(img_tensor, k, [1, 2])

            # Horizontal Flip
            if torch.rand(1).item() < 0.5:
                img_tensor = torch.flip(img_tensor, [2])

            # No Vertical Flip as per design

        # 6. Return
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


def get_loaders(load_cached_data=True, batch_size=config.BATCH_SIZE, fold_idx=None):
    """
    Generates DataLoaders.
    If fold_idx is None, uses fixed metadata splits.
    If fold_idx is provided, uses StratifiedKFold on the full training set (Cite Lesson 00052).
    """
    # 1. Load Raw Data
    raw_data = process_and_cache_data(load_cached_data)
    all_ids = raw_data["ids"]
    all_images = raw_data["images"]
    all_angles = raw_data["angles"]
    all_labels = raw_data["labels"]

    # Identify Test set
    test_mask = all_labels == -1
    test_indices = np.where(test_mask)[0]

    # Identify Train set (Full available training data)
    full_train_mask = ~test_mask
    full_train_indices = np.where(full_train_mask)[0]

    # Determine Train/Val indices
    if fold_idx is not None:
        # Dynamic K-Fold Split
        skf = StratifiedKFold(
            n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
        )

        # We need to split based on labels
        full_train_labels = all_labels[full_train_indices]

        splits = list(skf.split(full_train_indices, full_train_labels))
        train_idx_rel, val_idx_rel = splits[fold_idx]

        train_indices = full_train_indices[train_idx_rel]
        val_indices = full_train_indices[val_idx_rel]
    else:
        # Fixed Metadata Split (Fallback)
        id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}
        train_meta = pd.read_csv(config.TRAIN_META_CSV)
        val_meta = pd.read_csv(config.VAL_META_CSV)
        train_indices = [id_to_idx[i] for i in train_meta["id"].values]
        val_indices = [id_to_idx[i] for i in val_meta["id"].values]

    # 3. Impute Incidence Angle
    # Compute mean from TRAIN set only
    train_angles = all_angles[train_indices]
    angle_mean = np.nanmean(train_angles)
    all_angles = np.nan_to_num(all_angles, nan=angle_mean)

    # 4. Calculate Normalization Statistics
    # Compute stats on TRAIN set only
    train_imgs = all_images[train_indices]
    train_b1 = train_imgs[:, :, :, 0]
    train_b2 = train_imgs[:, :, :, 1]
    train_b3 = (train_b1 + train_b2) / 2.0

    stats = {
        "min": np.array(
            [np.min(train_b1), np.min(train_b2), np.min(train_b3)], dtype=np.float32
        ),
        "max": np.array(
            [np.max(train_b1), np.max(train_b2), np.max(train_b3)], dtype=np.float32
        ),
    }

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        images=all_images[train_indices],
        angles=all_angles[train_indices],
        labels=all_labels[train_indices],
        stats=stats,
        transform=True,
    )

    val_dataset = IcebergDataset(
        images=all_images[val_indices],
        angles=all_angles[val_indices],
        labels=all_labels[val_indices],
        stats=stats,
        transform=False,
    )

    test_dataset = IcebergDataset(
        images=all_images[test_indices],
        angles=all_angles[test_indices],
        labels=None,
        stats=stats,
        transform=False,
    )

    # 6. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
