import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into 3-channel images, applies normalization,
    and caches the result. If cache exists and load_cached_data is True, loads from cache.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)

    if load_cached_data and os.path.exists(Config.PROCESSED_DATA_PATH):
        # print(f"Loading cached data from {Config.PROCESSED_DATA_PATH}")
        try:
            data = np.load(Config.PROCESSED_DATA_PATH, allow_pickle=True)
            return (
                data["X_train"],
                data["y_train"],
                data["angles_train"],
                data["ids_train"],
                data["X_test"],
                data["angles_test"],
                data["ids_test"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # print("Processing data from scratch...")

    # Load Train Data
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)

    # Load Test Data
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # Helper function to process raw list of dicts into arrays
    def process_json_data(data_list, is_train=True):
        ids = []
        band_1 = []
        band_2 = []
        angles = []
        labels = []

        for item in data_list:
            ids.append(item["id"])
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            band_1.append(b1)
            band_2.append(b2)

            # Handle incidence angle
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            if is_train:
                labels.append(item["is_iceberg"])

        # Stack bands
        X_b1 = np.stack(band_1)
        X_b2 = np.stack(band_2)

        # Create 3rd channel: Average
        X_avg = (X_b1 + X_b2) / 2.0

        # Stack into (N, 75, 75, 3) -> PyTorch will want (N, 3, 75, 75) later
        # Keeping as (N, H, W, C) for easier normalization logic first
        X = np.stack([X_b1, X_b2, X_avg], axis=-1)

        angles = np.array(angles)
        ids = np.array(ids)

        if is_train:
            labels = np.array(labels)
            return X, angles, ids, labels
        else:
            return X, angles, ids, None

    # Process Train and Test
    X_train_raw, angles_train, ids_train, y_train = process_json_data(
        train_data, is_train=True
    )
    X_test_raw, angles_test, ids_test, _ = process_json_data(test_data, is_train=False)

    # --- Imputation for Incidence Angle ---
    # Calculate mean from training data (ignoring NaNs)
    angle_mean = np.nanmean(angles_train)

    # Fill NaNs
    angles_train = np.where(np.isnan(angles_train), angle_mean, angles_train)
    angles_test = np.where(np.isnan(angles_test), angle_mean, angles_test)

    # --- Normalization ---
    # Independent Per-Channel Min-Max Scaling based on Training Set
    # X shape: (N, 75, 75, 3)

    # Initialize arrays
    X_train = np.zeros_like(X_train_raw, dtype=np.float32)
    X_test = np.zeros_like(X_test_raw, dtype=np.float32)

    for i in range(3):  # For each channel
        # Calculate stats on training set
        train_ch = X_train_raw[:, :, :, i]
        ch_min = train_ch.min()
        ch_max = train_ch.max()
        denom = ch_max - ch_min + 1e-8  # Avoid div by zero

        # Apply to train
        X_train[:, :, :, i] = (train_ch - ch_min) / denom

        # Apply to test
        test_ch = X_test_raw[:, :, :, i]
        X_test[:, :, :, i] = (test_ch - ch_min) / denom

    # Transpose to (N, C, H, W) for PyTorch
    X_train = X_train.transpose(0, 3, 1, 2)
    X_test = X_test.transpose(0, 3, 1, 2)

    # Save to cache
    np.savez(
        Config.PROCESSED_DATA_PATH,
        X_train=X_train,
        y_train=y_train,
        angles_train=angles_train,
        ids_train=ids_train,
        X_test=X_test,
        angles_test=angles_test,
        ids_test=ids_test,
    )
    # print(f"Data processed and saved to {Config.PROCESSED_DATA_PATH}")

    return X_train, y_train, angles_train, ids_train, X_test, angles_test, ids_test


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg/Ship classification.
    Handles on-the-fly augmentation.
    """

    def __init__(self, images, angles, labels=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            transform (bool): Whether to apply augmentations
        """
        self.images = torch.tensor(images, dtype=torch.float32)
        self.angles = torch.tensor(angles, dtype=torch.float32)
        self.labels = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        angle = self.angles[idx]

        if self.transform:
            # Random Horizontal Flip
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[2])  # dims=[2] is width for (C, H, W)

            # Random Rotation (0, 90, 180, 270)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img = torch.rot90(img, k, dims=[1, 2])  # dims=[1, 2] are H, W

        if self.labels is not None:
            return img, angle, self.labels[idx]
        else:
            return img, angle, torch.tensor(-1.0)  # Placeholder for test set


def get_kfold_loaders(load_cached_data=True):
    """
    Generates Stratified K-Fold DataLoaders.

    Returns:
        list of (train_loader, val_loader) tuples.
    """
    X_train, y_train, angles_train, ids_train, _, _, _ = process_and_cache_data(
        load_cached_data
    )

    # Debug subset
    if Config.DEBUG:
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        angles_train = angles_train[indices]
        ids_train = ids_train[indices]

    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    loaders = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        # Split data
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        ang_tr, ang_val = angles_train[train_idx], angles_train[val_idx]

        # Create Datasets
        # Augmentation only for training
        train_dataset = IcebergDataset(
            X_tr, ang_tr, y_tr, transform=Config.USE_AUGMENTATION
        )
        val_dataset = IcebergDataset(X_val, ang_val, y_val, transform=False)

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        loaders.append((train_loader, val_loader))

    return loaders


def get_test_loader(load_cached_data=True):
    """
    Generates DataLoader for the test set.
    """
    _, _, _, _, X_test, angles_test, ids_test = process_and_cache_data(load_cached_data)

    test_dataset = IcebergDataset(X_test, angles_test, labels=None, transform=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader, ids_test
