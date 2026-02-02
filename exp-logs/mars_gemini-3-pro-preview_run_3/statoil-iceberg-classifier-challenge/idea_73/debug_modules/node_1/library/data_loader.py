import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from library.config import Config, IcebergDataset
from library.utils import impute_inc_angles


def load_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into 3-channel images (HH, HV, Avg),
    extracts labels and incidence angles, and caches the result as numpy arrays.

    Returns:
        tuple: (X_train, y_train, angle_train, ids_train, X_test, angle_test, ids_test)
               Note: angle_train and angle_test may contain NaNs.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(Config.WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.WORKING_DIR, "y_train.npy"),
        "angle_train": os.path.join(Config.WORKING_DIR, "angle_train.npy"),
        "ids_train": os.path.join(Config.WORKING_DIR, "ids_train.npy"),
        "X_test": os.path.join(Config.WORKING_DIR, "X_test.npy"),
        "angle_test": os.path.join(Config.WORKING_DIR, "angle_test.npy"),
        "ids_test": os.path.join(Config.WORKING_DIR, "ids_test.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        angle_train = np.load(cache_files["angle_train"])
        ids_train = np.load(cache_files["ids_train"], allow_pickle=True)
        X_test = np.load(cache_files["X_test"])
        angle_test = np.load(cache_files["angle_test"])
        ids_test = np.load(cache_files["ids_test"], allow_pickle=True)
        return X_train, y_train, angle_train, ids_train, X_test, angle_test, ids_test

    print("Processing data from scratch...")

    # Load raw JSON files
    train_path = os.path.join(Config.INPUT_DIR, Config.TRAIN_JSON)
    test_path = os.path.join(Config.INPUT_DIR, Config.TEST_JSON)

    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)

    def process_json_list(data_list, is_train=True):
        X = []
        angles = []
        ids = []
        y = []

        for item in data_list:
            # ID
            ids.append(item["id"])

            # Bands: Flattened list -> 75x75
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            # Synthetic 3rd channel: Average
            b3 = (b1 + b2) / 2.0

            # Stack to (3, 75, 75)
            img = np.stack([b1, b2, b3], axis=0)
            X.append(img)

            # Incidence Angle: 'na' -> NaN
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))

            # Target
            if is_train:
                y.append(item["is_iceberg"])

        X = np.array(X, dtype=np.float32)
        angles = np.array(angles, dtype=np.float32)
        ids = np.array(ids)
        y = np.array(y, dtype=np.float32) if is_train else None

        return X, angles, ids, y

    # Process Train
    X_train, angle_train, ids_train, y_train = process_json_list(
        train_data, is_train=True
    )
    # Process Test
    X_test, angle_test, ids_test, _ = process_json_list(test_data, is_train=False)

    # Save to cache
    print("Saving processed data to cache...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["ids_train"], ids_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angle_test"], angle_test)
    np.save(cache_files["ids_test"], ids_test)

    return X_train, y_train, angle_train, ids_train, X_test, angle_test, ids_test


def get_dataloaders(X, y, angles, train_idx, val_idx):
    """
    Creates training and validation DataLoaders for a specific fold.
    Performs leak-free imputation of incidence angles using the training fold median.

    Args:
        X (np.ndarray): Full training images (N, 3, 75, 75).
        y (np.ndarray): Full training labels (N,).
        angles (np.ndarray): Full training incidence angles (N,), potentially with NaNs.
        train_idx (np.ndarray): Indices for the training subset.
        val_idx (np.ndarray): Indices for the validation subset.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angle_train_raw = angles[train_idx]
    angle_val_raw = angles[val_idx]

    # Impute angles: Calculate median on train, apply to train and val
    angle_train_imp, angle_val_imp = impute_inc_angles(angle_train_raw, angle_val_raw)

    # Create Datasets
    # Transform=True for training (Horizontal/Vertical Flips)
    train_ds = IcebergDataset(X_train, angle_train_imp, y_train, transform=True)
    # Transform=False for validation
    val_ds = IcebergDataset(X_val, angle_val_imp, y_val, transform=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader


def get_test_loader(X_test, angle_test, angle_train_full):
    """
    Creates a DataLoader for the test set.
    Imputes missing test angles using the median of the full training set.

    Args:
        X_test (np.ndarray): Test images.
        angle_test (np.ndarray): Test angles (with NaNs).
        angle_train_full (np.ndarray): Full training angles (for calculating median).

    Returns:
        DataLoader: Test data loader.
    """
    # Impute test angles using full training set median
    # impute_inc_angles returns a tuple if multiple args are passed
    _, angle_test_imp = impute_inc_angles(angle_train_full, angle_test)

    test_ds = IcebergDataset(X_test, angle_test_imp, y=None, transform=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader
