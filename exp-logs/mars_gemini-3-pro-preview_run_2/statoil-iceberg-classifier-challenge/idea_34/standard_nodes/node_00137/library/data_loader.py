import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

# Import configuration from the provided library
from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    CACHE_PATH,
    SEED,
    BATCH_SIZE,
    NUM_FOLDS,
    IMAGE_SIZE,
    NUM_CHANNELS,
)


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            transform (bool): Whether to apply geometric augmentations.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image: (3, 75, 75)
        img = self.images[idx]
        angle = self.angles[idx]

        # Augmentation
        if self.transform:
            # Random Rotation (0, 90, 180, 270 degrees)
            # k is number of 90 degree rotations
            k = np.random.randint(0, 4)
            img = np.rot90(img, k, axes=(1, 2))

            # Random Horizontal Flip
            # axis 2 is width in (C, H, W) format
            if np.random.random() > 0.5:
                img = np.flip(img, axis=2)

            # Note: Vertical Flip is excluded as per instructions

        # Convert to tensor and ensure float32
        # Copy is necessary because numpy strides might be negative after flip/rot
        img_tensor = torch.from_numpy(img.copy()).float()
        angle_tensor = torch.tensor([angle], dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor([self.labels[idx]], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


def process_and_cache_data(load_cached_data=True):
    """
    Loads JSON data, processes it into tensors, normalizes globally, and caches it.

    Returns:
        X_train (np.ndarray): Normalized training images
        y_train (np.ndarray): Training labels
        inc_train (np.ndarray): Training incidence angles
        X_test (np.ndarray): Normalized test images
        inc_test (np.ndarray): Test incidence angles
        ids_test (np.ndarray): Test IDs
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    if load_cached_data and os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}...")
        try:
            data = np.load(CACHE_PATH)
            return (
                data["X_train"],
                data["y_train"],
                data["inc_train"],
                data["X_test"],
                data["inc_test"],
                data["ids_test"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # Load JSONs
    with open(TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(TEST_JSON, "r") as f:
        test_data = json.load(f)

    # Helper to process list of dicts
    def process_json(data_list, is_train=True, mean_angle=None):
        images = []
        angles = []
        ids = []
        labels = []

        # Calculate mean angle from valid data if not provided (for imputation)
        if mean_angle is None:
            valid_angles = [x["inc_angle"] for x in data_list if x["inc_angle"] != "na"]
            mean_angle = np.mean(valid_angles) if valid_angles else 0.0

        for item in data_list:
            # Extract Bands
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            # Channel 3: Arithmetic Mean of Band 1 and Band 2
            b3 = (b1 + b2) / 2.0

            # Stack into (3, 75, 75)
            img = np.stack([b1, b2, b3], axis=0)
            images.append(img)

            # Handle Incidence Angle
            ang = item["inc_angle"]
            if ang == "na":
                ang = mean_angle
            angles.append(float(ang))

            ids.append(item["id"])

            if is_train:
                labels.append(item["is_iceberg"])

        return (
            np.array(images, dtype=np.float32),
            np.array(angles, dtype=np.float32),
            np.array(ids),
            np.array(labels, dtype=np.float32) if is_train else None,
            mean_angle,
        )

    # Process Train
    print("Parsing train.json...")
    X_train_raw, inc_train_raw, _, y_train, train_mean_angle = process_json(
        train_data, is_train=True
    )

    # Process Test
    print("Parsing test.json...")
    # Use training mean angle for test imputation to ensure consistency and prevent leakage
    X_test_raw, inc_test_raw, ids_test, _, _ = process_json(
        test_data, is_train=False, mean_angle=train_mean_angle
    )

    # Global Normalization
    # Compute stats on Training set ONLY to avoid data leakage
    print("Computing global stats...")

    mins = []
    maxs = []
    for c in range(3):
        channel_data = X_train_raw[:, c, :, :]
        mins.append(np.min(channel_data))
        maxs.append(np.max(channel_data))

    print(f"Global Mins: {mins}")
    print(f"Global Maxs: {maxs}")

    # Apply normalization function
    def normalize(X, mins, maxs):
        X_norm = np.zeros_like(X)
        for c in range(3):
            # Min-Max Scaling
            denom = maxs[c] - mins[c]
            if denom == 0:
                denom = 1.0
            # Note: We do NOT clip values. Outliers in test set are preserved.
            X_norm[:, c, :, :] = (X[:, c, :, :] - mins[c]) / denom
        return X_norm

    X_train = normalize(X_train_raw, mins, maxs)
    X_test = normalize(X_test_raw, mins, maxs)

    # Cache the processed data
    print(f"Saving processed data to {CACHE_PATH}...")
    np.savez(
        CACHE_PATH,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train_raw,
        X_test=X_test,
        inc_test=inc_test_raw,
        ids_test=ids_test,
    )

    print("Data processed and cached.")
    return X_train, y_train, inc_train_raw, X_test, inc_test_raw, ids_test


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Returns train and validation DataLoaders for a specific fold in Stratified K-Fold.

    Args:
        fold_idx (int): The index of the fold (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        train_loader, val_loader
    """
    # Load data
    X_train, y_train, inc_train, _, _, _ = process_and_cache_data(load_cached_data)

    # Create Stratified K-Fold
    # We use the fixed SEED to ensure that the splits are identical across different runs/calls
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Generate splits
    splits = list(skf.split(X_train, y_train))
    if fold_idx >= len(splits):
        raise ValueError(f"Fold index {fold_idx} out of range for {NUM_FOLDS} folds.")

    train_idx, val_idx = splits[fold_idx]

    # Subset data
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    inc_tr, inc_val = inc_train[train_idx], inc_train[val_idx]

    # Create Datasets
    # Apply augmentation (transform=True) only to the training set
    train_ds = IcebergDataset(X_tr, inc_tr, y_tr, transform=True)
    val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)

    # Create Loaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns test DataLoader and test IDs.

    Returns:
        test_loader, ids_test
    """
    _, _, _, X_test, inc_test, ids_test = process_and_cache_data(load_cached_data)

    # No augmentation for test set
    test_ds = IcebergDataset(X_test, inc_test, labels=None, transform=False)

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    return test_loader, ids_test
