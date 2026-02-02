import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    CACHE_DIR,
    SEED,
    NUM_WORKERS,
    BATCH_SIZE,
    DEBUG,
    DEBUG_SUBSET_SIZE,
    NUM_FOLDS,
)
from library.utils import set_seed


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    Constructs a 3-channel image (HH, HV, Avg) from the 2-band input.
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Input images of shape (N, 2, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Targets of shape (N,).
            ids (np.ndarray, optional): Image IDs of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve bands (2, 75, 75)
        img_bands = self.X[idx]

        # Construct 3rd channel: Average of Band 1 and Band 2
        # Shape: (3, 75, 75)
        band1 = img_bands[0]
        band2 = img_bands[1]
        avg = (band1 + band2) / 2.0

        # Stack to create 3-channel image
        img_np = np.stack([band1, band2, avg], axis=0)

        # Convert to tensor
        img_tensor = torch.from_numpy(img_np).float()

        # Apply augmentations if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Prepare angle
        angle_tensor = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.y is not None:
            # Training/Validation mode
            label_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            # Test mode
            id_str = self.ids[idx]
            return img_tensor, angle_tensor, id_str


def get_transforms(train=True):
    """
    Returns the data augmentation pipeline.

    Args:
        train (bool): If True, apply random flips.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    return None


def _process_json_to_numpy(json_path, is_train=True):
    """
    Helper to parse raw JSON and convert to numpy arrays.
    """
    # print(f"Loading raw data from {json_path}...")
    # Using pandas for easier json parsing, though json module is also fine
    df = pd.read_json(json_path)

    # Process Images
    # Flattened lists to (N, 75, 75)
    band_1 = np.array([np.array(b).reshape(75, 75) for b in df["band_1"]])
    band_2 = np.array([np.array(b).reshape(75, 75) for b in df["band_2"]])

    # Stack to (N, 2, 75, 75)
    X = np.stack([band_1, band_2], axis=1).astype(np.float32)

    # Process Angles
    # Coerce 'na' to NaN
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    ids = df["id"].values

    if is_train:
        y = df["is_iceberg"].values.astype(np.float32)
        return X, angles, y, ids
    else:
        return X, angles, ids


def _load_cached_data(mode, load_cache=True):
    """
    Loads data from cache or processes from raw JSON if cache is missing/disabled.

    Args:
        mode (str): 'train' or 'test'.
        load_cache (bool): Whether to attempt loading from cache.

    Returns:
        Tuple of numpy arrays.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache paths
    cache_X = os.path.join(CACHE_DIR, f"X_{mode}.npy")
    cache_angle = os.path.join(CACHE_DIR, f"angle_{mode}.npy")
    cache_ids = os.path.join(CACHE_DIR, f"ids_{mode}.npy")
    cache_y = os.path.join(CACHE_DIR, f"y_{mode}.npy") if mode == "train" else None

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_X)
        and os.path.exists(cache_angle)
        and os.path.exists(cache_ids)
    )
    if mode == "train":
        cache_exists = cache_exists and os.path.exists(cache_y)

    if load_cache and cache_exists:
        # print(f"Loading {mode} data from cache...")
        X = np.load(cache_X)
        angles = np.load(cache_angle)
        ids = np.load(cache_ids, allow_pickle=True)
        if mode == "train":
            y = np.load(cache_y)
            return X, angles, y, ids
        return X, angles, ids

    # Process from scratch
    json_path = TRAIN_JSON if mode == "train" else TEST_JSON
    if mode == "train":
        X, angles, y, ids = _process_json_to_numpy(json_path, is_train=True)
        np.save(cache_X, X)
        np.save(cache_angle, angles)
        np.save(cache_y, y)
        np.save(cache_ids, ids)
        return X, angles, y, ids
    else:
        X, angles, ids = _process_json_to_numpy(json_path, is_train=False)
        np.save(cache_X, X)
        np.save(cache_angle, angles)
        np.save(cache_ids, ids)
        return X, angles, ids


def get_data_loaders(fold_idx, load_cached_data=True):
    """
    Creates train and validation DataLoaders for a specific fold.
    Implements leak-free incidence angle imputation.

    Args:
        fold_idx (int): Current fold index (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        train_loader, val_loader
    """
    set_seed(SEED)

    # 1. Load all training data
    X, angles, y, ids = _load_cached_data("train", load_cached_data)

    # Debugging subset
    if DEBUG:
        subset_size = min(len(X), DEBUG_SUBSET_SIZE)
        X = X[:subset_size]
        angles = angles[:subset_size]
        y = y[:subset_size]
        ids = ids[:subset_size]

    # 2. Stratified K-Fold Split
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Get indices for the requested fold
    # list(skf.split) returns a list of (train_idx, val_idx)
    splits = list(skf.split(X, y))
    train_idx, val_idx = splits[fold_idx]

    # 3. Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    ids_train, ids_val = ids[train_idx], ids[val_idx]

    # Raw angles (contain NaNs)
    angles_train_raw = angles[train_idx]
    angles_val_raw = angles[val_idx]

    # 4. Leak-Free Imputation
    # Calculate median ONLY on training data for this fold
    angle_median = np.nanmedian(angles_train_raw)

    # Fill NaNs in both sets using the training median
    angles_train = np.nan_to_num(angles_train_raw, nan=angle_median)
    angles_val = np.nan_to_num(angles_val_raw, nan=angle_median)

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, ids_train, transform=get_transforms(train=True)
    )
    val_dataset = IcebergDataset(
        X_val, angles_val, y_val, ids_val, transform=get_transforms(train=False)
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates the test DataLoader.
    Imputes missing incidence angles using the median from the full training set.
    """
    # 1. Load Test Data
    X_test, angles_test_raw, ids_test = _load_cached_data("test", load_cached_data)

    # 2. Load Train Data (lightweight, just for angle median)
    # We need the global training median to impute test values consistently
    _, angles_train_all, _, _ = _load_cached_data("train", load_cached_data)

    if DEBUG:
        subset_size = min(len(X_test), DEBUG_SUBSET_SIZE)
        X_test = X_test[:subset_size]
        angles_test_raw = angles_test_raw[:subset_size]
        ids_test = ids_test[:subset_size]

    # 3. Impute Test Angles
    # Use median of all training data
    angle_median = np.nanmedian(angles_train_all)
    angles_test = np.nan_to_num(angles_test_raw, nan=angle_median)

    # 4. Create Dataset
    test_dataset = IcebergDataset(
        X_test, angles_test, y=None, ids=ids_test, transform=get_transforms(train=False)
    )

    # 5. Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return test_loader
