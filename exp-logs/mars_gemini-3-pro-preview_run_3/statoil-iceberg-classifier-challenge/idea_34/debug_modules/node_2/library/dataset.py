import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles 3-channel image data and incidence angles.
    """

    def __init__(self, X, angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 75, 75, 3).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve image and angle
        img = self.X[idx]  # (75, 75, 3)
        angle = self.angles[idx]

        # Convert to Tensor and permute to (C, H, W)
        # Input is float32, keep it that way
        img_tensor = torch.from_numpy(img).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # (3, 75, 75)

        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply augmentations if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


def _process_json_data(file_path, is_train=True, angle_imputer=None):
    """
    Helper function to process raw JSON data into numpy arrays.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Process Images
    # Band 1 (HH) and Band 2 (HV) are lists of 5625 floats
    b1 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
    )
    b2 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
    )

    # Band 3 is the average of Band 1 and Band 2
    b3 = (b1 + b2) / 2.0

    # Stack to create (N, 75, 75, 3)
    # Stack along the last axis
    X = np.stack([b1, b2, b3], axis=-1)

    # Process Angles
    # Replace 'na' with NaN and convert to float
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    # Impute missing angles
    if is_train:
        # Calculate median from valid training data
        valid_angles = angles[~np.isnan(angles)]
        median_angle = np.median(valid_angles)
        angle_imputer = median_angle

    # Fill NaNs with the imputer value (train median)
    if angle_imputer is not None:
        angles[np.isnan(angles)] = angle_imputer
    else:
        # Fallback if no imputer provided (should not happen in this pipeline)
        angles[np.isnan(angles)] = 0.0

    ids = df["id"].values

    if is_train:
        y = df["is_iceberg"].values.astype(np.float32)
        return X, angles, y, ids, angle_imputer
    else:
        return X, angles, ids


def load_data(load_cached_data=True):
    """
    Loads data from cache or processes raw files if cache is missing.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        tuple: (X_train, y_train, angle_train, X_test, ids_test, angle_test)
    """
    # Define required cache keys
    required_keys = [
        "X_train",
        "y_train",
        "angle_train",
        "X_test",
        "ids_test",
        "angle_test",
    ]
    cache_exists = all(os.path.exists(Config.CACHE_PATHS[k]) for k in required_keys)

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = np.load(Config.CACHE_PATHS["X_train"])
        y_train = np.load(Config.CACHE_PATHS["y_train"])
        angle_train = np.load(Config.CACHE_PATHS["angle_train"])
        X_test = np.load(Config.CACHE_PATHS["X_test"])
        ids_test = np.load(Config.CACHE_PATHS["ids_test"])
        angle_test = np.load(Config.CACHE_PATHS["angle_test"])
    else:
        print("Processing raw data...")
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Process Train
        X_train, angle_train, y_train, _, median_angle = _process_json_data(
            Config.TRAIN_JSON, is_train=True
        )

        # Process Test (using median_angle from train for imputation)
        X_test, angle_test, ids_test = _process_json_data(
            Config.TEST_JSON, is_train=False, angle_imputer=median_angle
        )

        # Save to cache
        np.save(Config.CACHE_PATHS["X_train"], X_train)
        np.save(Config.CACHE_PATHS["y_train"], y_train)
        np.save(Config.CACHE_PATHS["angle_train"], angle_train)
        np.save(Config.CACHE_PATHS["X_test"], X_test)
        np.save(Config.CACHE_PATHS["ids_test"], ids_test)
        np.save(Config.CACHE_PATHS["angle_test"], angle_test)

        print("Data processed and cached.")

    return X_train, y_train, angle_train, X_test, ids_test, angle_test


def get_data_loaders(fold_idx=0, load_cached_data=True):
    """
    Generates DataLoaders for a specific fold in Stratified K-Fold.

    Args:
        fold_idx (int): The fold index (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # 1. Load all data
    X_train_all, y_train_all, angle_train_all, X_test, ids_test, angle_test = load_data(
        load_cached_data
    )

    # 2. Debug Mode: Subset data if enabled
    if Config.DEBUG:
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(X_train_all))
        indices = np.random.choice(len(X_train_all), subset_size, replace=False)
        X_train_all = X_train_all[indices]
        y_train_all = y_train_all[indices]
        angle_train_all = angle_train_all[indices]

        subset_size_test = min(Config.DEBUG_SUBSET_SIZE, len(X_test))
        indices_test = np.random.choice(len(X_test), subset_size_test, replace=False)
        X_test = X_test[indices_test]
        ids_test = ids_test[indices_test]
        angle_test = angle_test[indices_test]
        print(f"DEBUG mode: Reduced training set to {len(X_train_all)} samples.")

    # 3. Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    # list(skf.split) returns a list of (train_idx, val_idx) tuples
    splits = list(skf.split(X_train_all, y_train_all))
    if fold_idx >= len(splits):
        raise ValueError(
            f"Fold index {fold_idx} out of range for {Config.NUM_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_idx]

    # Subset the data
    X_train_fold = X_train_all[train_idx]
    y_train_fold = y_train_all[train_idx]
    angle_train_fold = angle_train_all[train_idx]

    X_val_fold = X_train_all[val_idx]
    y_val_fold = y_train_all[val_idx]
    angle_val_fold = angle_train_all[val_idx]

    # 4. Define Transforms
    # Augmentations for training only
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # No augmentations for validation/test
    val_transform = None

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        X_train_fold, angle_train_fold, y_train_fold, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val_fold, angle_val_fold, y_val_fold, transform=val_transform
    )
    test_dataset = IcebergDataset(X_test, angle_test, transform=val_transform)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
