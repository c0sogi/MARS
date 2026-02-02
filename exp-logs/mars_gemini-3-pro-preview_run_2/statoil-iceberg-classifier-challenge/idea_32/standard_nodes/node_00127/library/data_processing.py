import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config

# Set fixed random seeds for reproducibility
import random

random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


def compute_global_stats(images):
    """
    Compute global min and max for each channel across the entire dataset.
    images: (N, 75, 75, 3)
    Returns: min_vals (3,), max_vals (3,)
    """
    # Reshape to (Total_Pixels, 3) to compute stats per channel
    reshaped = images.reshape(-1, 3)
    min_vals = np.min(reshaped, axis=0)
    max_vals = np.max(reshaped, axis=0)
    return min_vals, max_vals


def process_json_data(json_path, ids_filter=None):
    """
    Load json and process into numpy arrays.
    Constructs the 3rd channel (Mean of Band 1 and Band 2).
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Filter data based on IDs if a filter is provided
    if ids_filter is not None:
        data = [d for d in data if d["id"] in ids_filter]

    if not data:
        return np.array([]), np.array([]), np.array([]), np.array([])

    ids = [d["id"] for d in data]

    # Extract bands and reshape to (N, 75, 75)
    band_1 = np.array([d["band_1"] for d in data]).reshape(-1, 75, 75)
    band_2 = np.array([d["band_2"] for d in data]).reshape(-1, 75, 75)

    # Construct 3rd channel: Arithmetic Mean
    band_3 = (band_1 + band_2) / 2.0

    # Stack to create (N, 75, 75, 3)
    images = np.stack([band_1, band_2, band_3], axis=-1)

    # Extract incidence angles, handling 'na'
    inc_angles = []
    for d in data:
        ia = d["inc_angle"]
        if ia == "na":
            inc_angles.append(np.nan)
        else:
            inc_angles.append(float(ia))
    inc_angles = np.array(inc_angles)

    # Extract labels if they exist
    labels = None
    if "is_iceberg" in data[0]:
        labels = np.array([d["is_iceberg"] for d in data])

    return ids, images, inc_angles, labels


def load_data(load_cached_data=True):
    """
    Loads data. Uses cache if available and requested.
    Otherwise, processes raw JSONs, computes stats, and saves to cache.
    """
    Config.setup()

    if load_cached_data and os.path.exists(Config.PROCESSED_DATA_PATH):
        print(f"Loading cached data from {Config.PROCESSED_DATA_PATH}")
        try:
            loaded = np.load(Config.PROCESSED_DATA_PATH, allow_pickle=True)
            data_dict = {k: loaded[k] for k in loaded.files}
            # Cite debug_lesson_6: Validate Cached Data Schema Before Usage
            if "X_val" not in data_dict:
                raise ValueError("Cached data missing validation set.")
            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing from scratch...")

    print("Processing data from scratch...")

    # 1. Identify Data Splits from Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Cite debug_lesson_10: Enforce Predefined Data Splits by Filtering Raw Inputs
    train_ids_set = set(df_train_meta["id"].values)
    val_ids_set = set(df_val_meta["id"].values)
    test_ids_set = set(df_test_meta["id"].values)

    # 2. Process Raw JSONs
    # Train Split
    t_ids, t_imgs, t_inc, t_lbls = process_json_data(
        Config.TRAIN_JSON, ids_filter=train_ids_set
    )
    # Validation Split (Hold-out)
    v_ids, v_imgs, v_inc, v_lbls = process_json_data(
        Config.TRAIN_JSON, ids_filter=val_ids_set
    )
    # Test Split
    test_ids, test_imgs, test_inc, _ = process_json_data(
        Config.TEST_JSON, ids_filter=test_ids_set
    )

    # 3. Impute Missing Incidence Angles
    # Calculate mean from training data only
    train_inc_mean = np.nanmean(t_inc)

    # Impute NaNs using training mean to prevent leakage
    t_inc_imputed = np.where(np.isnan(t_inc), train_inc_mean, t_inc)
    v_inc_imputed = np.where(np.isnan(v_inc), train_inc_mean, v_inc)
    test_inc_imputed = np.where(np.isnan(test_inc), train_inc_mean, test_inc)

    # 4. Compute Global Normalization Statistics
    # Derived from the training dataset only
    min_vals, max_vals = compute_global_stats(t_imgs)

    # 5. Pack Data
    data_dict = {
        "X_train": t_imgs.astype(np.float32),
        "y_train": t_lbls.astype(np.int64),
        "inc_train": t_inc_imputed.astype(np.float32),
        "ids_train": np.array(t_ids),
        "X_val": v_imgs.astype(np.float32),
        "y_val": v_lbls.astype(np.int64),
        "inc_val": v_inc_imputed.astype(np.float32),
        "ids_val": np.array(v_ids),
        "X_test": test_imgs.astype(np.float32),
        "inc_test": test_inc_imputed.astype(np.float32),
        "ids_test": np.array(test_ids),
        "min_vals": min_vals.astype(np.float32),
        "max_vals": max_vals.astype(np.float32),
    }

    # 6. Save to Cache
    print(f"Saving processed data to {Config.PROCESSED_DATA_PATH}")
    np.savez(Config.PROCESSED_DATA_PATH, **data_dict)

    return data_dict


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles on-the-fly normalization and augmentation.
    """

    def __init__(
        self,
        images,
        inc_angles,
        labels=None,
        min_vals=None,
        max_vals=None,
        transform=False,
    ):
        self.images = images
        self.inc_angles = inc_angles
        self.labels = labels
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image: (75, 75, 3)
        img = self.images[idx]
        inc = self.inc_angles[idx]

        # 1. Global Normalization
        # (img - min) / (max - min)
        if self.min_vals is not None and self.max_vals is not None:
            denom = self.max_vals - self.min_vals
            # Prevent division by zero
            denom[denom == 0] = 1.0
            img = (img - self.min_vals) / denom
            # Note: No hard clipping applied

        # 2. Augmentation
        if self.transform:
            # Random Rotation: 0, 90, 180, 270 degrees
            k = np.random.randint(0, 4)
            img = np.rot90(img, k=k, axes=(0, 1))

            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img)

            # Vertical flip is excluded per instructions

        # 3. Convert to Tensor
        # Transpose from (H, W, C) to (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        # Create tensors
        img_tensor = torch.from_numpy(img.copy()).float()
        inc_tensor = torch.tensor(inc, dtype=torch.float32)

        if self.labels is not None:
            # BCEWithLogitsLoss expects float targets
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, inc_tensor, label_tensor
        else:
            return img_tensor, inc_tensor


def get_fold_loaders(fold_idx, load_cached_data=True):
    """
    Returns train and validation DataLoaders for a specific fold.
    Uses Stratified K-Fold on the full training dataset.
    """
    data = load_data(load_cached_data)
    X = data["X_train"]
    y = data["y_train"]
    inc = data["inc_train"]
    min_vals = data["min_vals"]
    max_vals = data["max_vals"]

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get splits generator
    splits = list(skf.split(X, y))

    if fold_idx < 0 or fold_idx >= len(splits):
        raise ValueError(
            f"Fold index {fold_idx} out of range for {Config.NUM_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_idx]

    # Create Datasets
    # Train: With Augmentation
    train_ds = IcebergDataset(
        X[train_idx], inc[train_idx], y[train_idx], min_vals, max_vals, transform=True
    )

    # Validation: No Augmentation
    val_ds = IcebergDataset(
        X[val_idx], inc[val_idx], y[val_idx], min_vals, max_vals, transform=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns the test DataLoader and the corresponding IDs.
    """
    data = load_data(load_cached_data)
    X = data["X_test"]
    inc = data["inc_test"]
    ids = data["ids_test"]
    min_vals = data["min_vals"]
    max_vals = data["max_vals"]

    # Test Dataset: No Augmentation
    test_ds = IcebergDataset(
        X, inc, labels=None, min_vals=min_vals, max_vals=max_vals, transform=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_loader, ids
