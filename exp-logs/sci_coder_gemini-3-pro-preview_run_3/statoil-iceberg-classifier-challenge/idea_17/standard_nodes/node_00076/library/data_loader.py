import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from library.config import Config
from library.utils import set_seed


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None):
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load data
        img = self.X[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle).float()

        # Apply augmentations (only works on Tensor or PIL)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Return tuple based on mode (Train/Val vs Test)
        if self.y is not None:
            label = torch.tensor(self.y[idx]).float()
            return img_tensor, angle_tensor, label
        else:
            img_id = self.ids[idx]
            return img_tensor, angle_tensor, img_id


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes bands/angles, and caches as .npy files.
    Returns the processed numpy arrays.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # File paths for cache
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "angles_train": os.path.join(Config.CACHE_DIR, "angles_train.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
        "angles_test": os.path.join(Config.CACHE_DIR, "angles_test.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        angles_train = np.load(cache_files["angles_train"])
        X_test = np.load(cache_files["X_test"])
        ids_test = np.load(cache_files["ids_test"])
        angles_test = np.load(cache_files["angles_test"])
        return X_train, angles_train, y_train, X_test, angles_test, ids_test

    print("Processing data from scratch...")

    # --- Process Training Data ---
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)

    # Pre-allocate arrays
    num_train = len(train_data)
    X_train = np.zeros((num_train, 3, 75, 75), dtype=np.float32)
    y_train = np.zeros(num_train, dtype=np.float32)
    angles_train = np.zeros(num_train, dtype=np.float32)

    train_angles_list = []

    for i, item in enumerate(train_data):
        # Band 1 (HH)
        b1 = np.array(item["band_1"]).reshape(75, 75)
        # Band 2 (HV)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        # Band 3 (Avg)
        b3 = (b1 + b2) / 2.0

        # Stack: (3, 75, 75)
        X_train[i, 0, :, :] = b1
        X_train[i, 1, :, :] = b2
        X_train[i, 2, :, :] = b3

        y_train[i] = item["is_iceberg"]

        # Angle handling
        ang = item["inc_angle"]
        if ang == "na":
            train_angles_list.append(np.nan)
        else:
            train_angles_list.append(float(ang))

    # --- Process Test Data ---
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    num_test = len(test_data)
    X_test = np.zeros((num_test, 3, 75, 75), dtype=np.float32)
    ids_test = []
    test_angles_list = []

    for i, item in enumerate(test_data):
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        b3 = (b1 + b2) / 2.0

        X_test[i, 0, :, :] = b1
        X_test[i, 1, :, :] = b2
        X_test[i, 2, :, :] = b3

        ids_test.append(item["id"])

        ang = item["inc_angle"]
        if ang == "na":
            test_angles_list.append(np.nan)
        else:
            test_angles_list.append(float(ang))

    ids_test = np.array(ids_test)

    # --- Impute Angles ---
    # Convert to numpy for easier handling
    angles_train_raw = np.array(train_angles_list)
    angles_test_raw = np.array(test_angles_list)

    # Compute median from training data (ignoring NaNs)
    angle_median = np.nanmedian(angles_train_raw)

    # Fill NaNs
    angles_train = np.where(
        np.isnan(angles_train_raw), angle_median, angles_train_raw
    ).astype(np.float32)
    angles_test = np.where(
        np.isnan(angles_test_raw), angle_median, angles_test_raw
    ).astype(np.float32)

    # --- Save to Cache ---
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angles_train"], angles_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["ids_test"], ids_test)
    np.save(cache_files["angles_test"], angles_test)

    print(f"Data processed and saved to {Config.CACHE_DIR}")
    return X_train, angles_train, y_train, X_test, angles_test, ids_test


def get_dataloaders(fold_idx=0, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold in the 5-Fold Cross Validation.
    Cite solution_lesson_node_00044: Use Hold-Out Ensemble strategy.

    Splits data into:
    1. Hold-Out Set (20%) - Fixed
    2. Development Set (80%) - Used for CV

    Args:
        fold_idx (int): Index of the validation fold (0 to NUM_FOLDS-1) within the Development Set.
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        train_loader, val_loader, holdout_loader, test_loader
    """
    set_seed(Config.SEED)

    # 1. Load Data
    X, angles, y, X_test, angles_test, ids_test = process_and_cache_data(
        load_cached_data
    )

    # Debugging: Subset data if enabled
    if Config.DEBUG:
        subset_size = min(len(y), Config.DEBUG_SUBSET_SIZE)
        X = X[:subset_size]
        angles = angles[:subset_size]
        y = y[:subset_size]

        test_subset = min(len(ids_test), Config.DEBUG_SUBSET_SIZE)
        X_test = X_test[:test_subset]
        angles_test = angles_test[:test_subset]
        ids_test = ids_test[:test_subset]
        print(f"DEBUG Mode: Truncated data to {subset_size} samples.")

    # 2. Create Fixed Hold-Out Split (20%)
    # Using Stratified Split to maintain class balance
    X_dev, X_holdout, angles_dev, angles_holdout, y_dev, y_holdout = train_test_split(
        X, angles, y, test_size=0.20, random_state=Config.SEED, stratify=y
    )

    # 3. Split Development Data (Stratified K-Fold)
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    splits = list(skf.split(X_dev, y_dev))
    if fold_idx < 0 or fold_idx >= Config.NUM_FOLDS:
        raise ValueError(f"fold_idx must be between 0 and {Config.NUM_FOLDS - 1}")

    train_idx, val_idx = splits[fold_idx]

    # Create CV subsets
    X_train_fold = X_dev[train_idx]
    angles_train_fold = angles_dev[train_idx]
    y_train_fold = y_dev[train_idx]

    X_val_fold = X_dev[val_idx]
    angles_val_fold = angles_dev[val_idx]
    y_val_fold = y_dev[val_idx]

    # 3. Define Transforms
    # Augmentation for training only
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # No augmentation for validation/test
    val_transform = None

    # 4. Create Datasets
    train_dataset = IcebergDataset(
        X_train_fold, angles_train_fold, y_train_fold, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val_fold, angles_val_fold, y_val_fold, transform=val_transform
    )
    holdout_dataset = IcebergDataset(
        X_holdout, angles_holdout, y_holdout, transform=val_transform
    )
    test_dataset = IcebergDataset(
        X_test, angles_test, ids=ids_test, transform=val_transform
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, holdout_loader, test_loader
