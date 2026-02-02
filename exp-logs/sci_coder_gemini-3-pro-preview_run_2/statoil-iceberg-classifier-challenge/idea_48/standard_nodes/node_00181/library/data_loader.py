import os
import json
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_JSON,
    TEST_JSON,
    CACHE_FILE,
    SEED,
    IMAGE_SIZE,
    BATCH_SIZE,
)
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_loader", os.path.join(WORKING_DIR, "data_loader.log"))


def get_global_stats(X):
    """
    Computes global min and max values for each channel across the entire dataset.

    Args:
        X (np.ndarray): Input images of shape (N, 3, H, W).

    Returns:
        tuple: (min_vals, max_vals) where each is a numpy array of shape (3,).
    """
    # Transpose to (N, H, W, 3) and flatten spatial dimensions -> (N*H*W, 3)
    pixels = X.transpose(0, 2, 3, 1).reshape(-1, 3)
    min_vals = pixels.min(axis=0)
    max_vals = pixels.max(axis=0)
    return min_vals, max_vals


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.

    Features:
    - Returns 3-channel images (Band 1, Band 2, Mean).
    - Handles incidence angles.
    - Implements specific augmentations: Random 90-degree rotations and Horizontal Flips.
    """

    def __init__(self, X, inc_angles, y=None, transform=False):
        self.X = X
        self.inc_angles = inc_angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # Shape: (3, 75, 75)
        inc = self.inc_angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()

        if self.transform:
            # Augmentation: Random Rotation (0, 90, 180, 270 degrees)
            k = random.randint(0, 3)
            img_tensor = torch.rot90(img_tensor, k, [1, 2])

            # Augmentation: Horizontal Flip (flip width dimension)
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])

        inc_tensor = torch.tensor([inc], dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.long)
            return img_tensor, inc_tensor, label
        else:
            return img_tensor, inc_tensor


def load_and_process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes bands, applies global normalization, and caches the result.

    Returns:
        tuple: (X_train, y_train, inc_train, train_ids, X_test, inc_test, test_ids)
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(CACHE_FILE):
        logger.info(f"Loading cached data from {CACHE_FILE}")
        try:
            data = np.load(CACHE_FILE, allow_pickle=True)
            return (
                data["X_train"],
                data["y_train"],
                data["inc_train"],
                data["train_ids"],
                data["X_test"],
                data["inc_test"],
                data["test_ids"],
            )
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    logger.info("Processing data from scratch...")

    # 2. Load Raw Data
    with open(TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(TEST_JSON, "r") as f:
        test_data = json.load(f)

    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # 3. Process Image Bands
    def process_images(df):
        # Extract bands and reshape to 75x75
        b1 = np.array([np.array(band).reshape(75, 75) for band in df["band_1"]])
        b2 = np.array([np.array(band).reshape(75, 75) for band in df["band_2"]])

        # Channel 3: Arithmetic Mean of Band 1 and Band 2
        b3 = (b1 + b2) / 2.0

        # Stack to (N, 3, 75, 75)
        images = np.stack([b1, b2, b3], axis=1).astype(np.float32)
        return images

    X_train = process_images(df_train)
    X_test = process_images(df_test)

    # 4. Extract Targets and IDs
    y_train = df_train["is_iceberg"].values.astype(np.int64)
    train_ids = df_train["id"].values
    test_ids = df_test["id"].values

    # 5. Process Incidence Angles
    # Calculate mean from training set to impute missing values
    train_inc_numeric = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    inc_mean = train_inc_numeric.mean()

    def process_inc_angle(series, fill_value):
        vals = pd.to_numeric(series, errors="coerce")
        vals = vals.fillna(fill_value)
        return vals.values.astype(np.float32)

    inc_train = process_inc_angle(df_train["inc_angle"], inc_mean)
    inc_test = process_inc_angle(df_test["inc_angle"], inc_mean)

    # 6. Global Normalization
    # Compute statistics on the ENTIRE training dataset
    logger.info("Computing global scaling statistics...")
    min_vals, max_vals = get_global_stats(X_train)
    logger.info(f"Global Min: {min_vals}, Global Max: {max_vals}")

    # Apply Min-Max scaling: (X - min) / (max - min)
    # No hard clipping (values can exceed [0, 1] in test set)
    for c in range(3):
        denom = max_vals[c] - min_vals[c] + 1e-8
        X_train[:, c, :, :] = (X_train[:, c, :, :] - min_vals[c]) / denom
        X_test[:, c, :, :] = (X_test[:, c, :, :] - min_vals[c]) / denom

    # 7. Save to Cache
    logger.info(f"Saving processed data to {CACHE_FILE}")
    np.savez(
        CACHE_FILE,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        train_ids=train_ids,
        X_test=X_test,
        inc_test=inc_test,
        test_ids=test_ids,
    )

    return X_train, y_train, inc_train, train_ids, X_test, inc_test, test_ids


def get_data_loaders(
    X_train, y_train, inc_train, X_val, y_val, inc_val, batch_size, num_workers
):
    """
    Creates DataLoaders for training and validation sets.

    Args:
        X_train, y_train, inc_train: Training data arrays.
        X_val, y_val, inc_val: Validation data arrays.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Training dataset with augmentation
    train_ds = IcebergDataset(X_train, inc_train, y_train, transform=True)

    # Validation dataset without augmentation
    val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(X_test, inc_test, batch_size, num_workers):
    """
    Creates a DataLoader for the test set.
    """
    test_ds = IcebergDataset(X_test, inc_test, y=None, transform=False)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
