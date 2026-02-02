import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_and_process_data(load_cached_data=True):
    """
    Loads raw data, processes it (reshaping, channel creation, normalization, imputation),
    and caches the result. If cache exists and load_cached_data is True, loads from cache.
    """
    cache_path = Config.CACHE_FILE

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading processed data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["X_train"],
                data["a_train"],
                data["y_train"],
                data["X_val"],
                data["a_val"],
                data["y_val"],
                data["X_test"],
                data["a_test"],
                data["test_ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch.")

    # 2. Process from Scratch
    print("Processing raw data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)
    test_meta = pd.read_csv(Config.TEST_META)

    # Load Raw JSONs
    print(f"Loading {Config.TRAIN_JSON}...")
    with open(Config.TRAIN_JSON, "r") as f:
        raw_train_data = json.load(f)

    print(f"Loading {Config.TEST_JSON}...")
    with open(Config.TEST_JSON, "r") as f:
        raw_test_data = json.load(f)

    # Create ID lookup dictionaries for raw data
    # raw_train_data contains both train and val samples (from the original train.json)
    raw_train_map = {item["id"]: item for item in raw_train_data}
    raw_test_map = {item["id"]: item for item in raw_test_data}

    # Helper to extract and reshape images
    def process_images(ids, source_map):
        images = []
        for img_id in ids:
            item = source_map[img_id]
            # Reshape to 75x75
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            # 3rd Channel: Average
            avg = (b1 + b2) / 2.0
            # Stack: (3, 75, 75)
            img = np.stack([b1, b2, avg], axis=0)
            images.append(img)
        return np.array(images, dtype=np.float32)

    # Helper to extract angles
    def process_angles(ids, source_map):
        angles = []
        for img_id in ids:
            angle_val = source_map[img_id]["inc_angle"]
            if angle_val == "na":
                angles.append(np.nan)
            else:
                angles.append(float(angle_val))
        return np.array(angles, dtype=np.float32)

    # Extract Data based on Metadata Splits
    print("Extracting tensors...")
    X_train = process_images(train_meta["id"].values, raw_train_map)
    a_train = process_angles(train_meta["id"].values, raw_train_map)
    y_train = train_meta["is_iceberg"].values.astype(np.float32)

    X_val = process_images(val_meta["id"].values, raw_train_map)
    a_val = process_angles(val_meta["id"].values, raw_train_map)
    y_val = val_meta["is_iceberg"].values.astype(np.float32)

    X_test = process_images(test_meta["id"].values, raw_test_map)
    a_test = process_angles(test_meta["id"].values, raw_test_map)
    test_ids = test_meta["id"].values

    # 3. Impute Missing Angles
    # Calculate mean from training set only
    angle_mean = np.nanmean(a_train)
    print(f"Imputing missing angles with training mean: {angle_mean:.4f}")

    # Fill NaNs
    a_train = np.nan_to_num(a_train, nan=angle_mean)
    a_val = np.nan_to_num(a_val, nan=angle_mean)
    a_test = np.nan_to_num(a_test, nan=angle_mean)

    # 4. Normalize Images (Independent Per-Channel Min-Max)
    print("Normalizing images...")
    # Calculate stats on training set
    # Shape of X_train is (N, 3, 75, 75)
    # We want min/max per channel (axis 0 is batch, axis 2,3 are spatial)
    # We flatten spatial dims to compute stats per channel

    # Channel 0
    c0_min = X_train[:, 0, :, :].min()
    c0_max = X_train[:, 0, :, :].max()

    # Channel 1
    c1_min = X_train[:, 1, :, :].min()
    c1_max = X_train[:, 1, :, :].max()

    # Channel 2
    c2_min = X_train[:, 2, :, :].min()
    c2_max = X_train[:, 2, :, :].max()

    print(
        f"Stats - Ch0: [{c0_min:.2f}, {c0_max:.2f}], Ch1: [{c1_min:.2f}, {c1_max:.2f}], Ch2: [{c2_min:.2f}, {c2_max:.2f}]"
    )

    def apply_minmax(X):
        X[:, 0, :, :] = (X[:, 0, :, :] - c0_min) / (c0_max - c0_min)
        X[:, 1, :, :] = (X[:, 1, :, :] - c1_min) / (c1_max - c1_min)
        X[:, 2, :, :] = (X[:, 2, :, :] - c2_min) / (c2_max - c2_min)
        return X

    X_train = apply_minmax(X_train)
    X_val = apply_minmax(X_val)
    X_test = apply_minmax(X_test)

    # 5. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_train=X_train,
        a_train=a_train,
        y_train=y_train,
        X_val=X_val,
        a_val=a_val,
        y_val=y_val,
        X_test=X_test,
        a_test=a_test,
        test_ids=test_ids,
    )

    return X_train, a_train, y_train, X_val, a_val, y_val, X_test, a_test, test_ids


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, ids=None, transform=False):
        """
        Args:
            images: Numpy array of shape (N, 3, 75, 75)
            angles: Numpy array of shape (N,)
            labels: Numpy array of shape (N,) or None
            ids: Array of strings or None
            transform: Boolean, whether to apply augmentation
        """
        self.images = torch.from_numpy(images).float()
        self.angles = torch.from_numpy(angles).float().unsqueeze(1)  # (N, 1)
        self.labels = (
            torch.from_numpy(labels).float().unsqueeze(1)
            if labels is not None
            else None
        )
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        angle = self.angles[idx]

        # Apply Augmentation
        if self.transform:
            # 1. Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            img = torch.rot90(img, k, dims=[1, 2])

            # 2. Random Horizontal Flip
            if np.random.random() > 0.5:
                img = torch.flip(img, dims=[2])

            # Note: Vertical flip excluded as per instructions

        if self.labels is not None:
            return img, angle, self.labels[idx]
        else:
            # For test set, return ID to track predictions
            return img, angle, self.ids[idx]


def get_loaders(load_cached_data=True):
    """
    Generates DataLoaders for Train, Validation, and Test sets.
    """
    # Load Data
    X_train, a_train, y_train, X_val, a_val, y_val, X_test, a_test, test_ids = (
        load_and_process_data(load_cached_data)
    )

    # Create Datasets
    # Train: Augmentation Enabled
    train_dataset = IcebergDataset(X_train, a_train, y_train, transform=True)

    # Val: No Augmentation
    val_dataset = IcebergDataset(X_val, a_val, y_val, transform=False)

    # Test: No Augmentation, include IDs
    test_dataset = IcebergDataset(
        X_test, a_test, labels=None, ids=test_ids, transform=False
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
