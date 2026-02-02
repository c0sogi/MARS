import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    Handles 3-channel image data, incidence angles, and labels.
    """

    def __init__(
        self, images, angles, targets=None, ids=None, transform=None, mode="train"
    ):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            targets (np.ndarray, optional): Shape (N,)
            ids (np.ndarray, optional): Shape (N,)
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
        """
        self.images = images
        self.angles = angles
        self.targets = targets
        self.ids = ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]
        angle = self.angles[idx]

        # Convert to tensor
        # Input image is numpy array (3, 75, 75), convert to float tensor
        image_tensor = torch.from_numpy(image).float()

        # Apply augmentations if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Convert angle to tensor
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.mode == "test":
            # For test set, return ID for submission generation
            id_val = self.ids[idx]
            return image_tensor, angle_tensor, id_val
        else:
            # For train/val, return label
            label = self.targets[idx]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return image_tensor, angle_tensor, label_tensor


def get_median_angle():
    """
    Calculates the median incidence angle from the training metadata.
    Used for imputing missing values in Train, Val, and Test sets.
    """
    df = pd.read_csv(Config.TRAIN_META_PATH)
    return df["inc_angle"].median()


def process_data(meta_path, raw_json_path, mode, angle_median, load_cached_data=True):
    """
    Loads raw JSON data, merges with metadata, processes images/angles, and caches results.

    Args:
        meta_path (str): Path to the metadata CSV file.
        raw_json_path (str): Path to the raw JSON file containing image bands.
        mode (str): 'train', 'val', or 'test'.
        angle_median (float): Value to use for imputing missing incidence angles.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X, angles, y, ids)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    cache_X = os.path.join(cache_dir, f"X_{mode}.npy")
    cache_angle = os.path.join(cache_dir, f"angle_{mode}.npy")
    cache_ids = os.path.join(cache_dir, f"ids_{mode}.npy")
    cache_y = os.path.join(cache_dir, f"y_{mode}.npy") if mode != "test" else None

    # Check if all required cache files exist
    files_exist = (
        os.path.exists(cache_X)
        and os.path.exists(cache_angle)
        and os.path.exists(cache_ids)
    )
    if mode != "test":
        files_exist = files_exist and os.path.exists(cache_y)

    # Load from cache if requested and available
    if load_cached_data and files_exist:
        print(f"Loading cached {mode} data from {cache_dir}...")
        X = np.load(cache_X)
        angles = np.load(cache_angle)
        ids = np.load(cache_ids, allow_pickle=True)
        y = np.load(cache_y) if mode != "test" else None
        return X, angles, y, ids

    print(f"Processing {mode} data from scratch (Source: {raw_json_path})...")

    # Load metadata and set index for joining
    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df.set_index("id")

    # Load raw JSON
    # Note: Loading entire JSON is memory intensive but fits within 220GB RAM.
    raw_df = pd.read_json(raw_json_path)
    raw_df = raw_df.set_index("id")

    # Join metadata with raw data to get the specific subset (train vs val)
    # Inner join ensures we only get rows present in the metadata split
    merged_df = meta_df.join(raw_df[["band_1", "band_2"]], how="inner")

    # Extract IDs (index)
    ids = merged_df.index.values

    # Process Images
    # Convert flattened lists to numpy arrays and reshape to (75, 75)
    # Band 1 (HH)
    b1 = np.stack([np.array(b) for b in merged_df["band_1"].values])
    b1 = b1.reshape(-1, 75, 75)

    # Band 2 (HV)
    b2 = np.stack([np.array(b) for b in merged_df["band_2"].values])
    b2 = b2.reshape(-1, 75, 75)

    # Band 3 (Average of HH and HV) - Synthetic channel
    b3 = (b1 + b2) / 2.0

    # Stack channels: (N, 3, 75, 75)
    X = np.stack([b1, b2, b3], axis=1)

    # Process Angles
    # Use 'inc_angle' from metadata (already has 'na' coerced to NaN)
    angles = merged_df["inc_angle"].values
    # Impute missing values with the training set median
    angles = np.nan_to_num(angles, nan=angle_median)

    # Process Targets
    y = None
    if mode != "test":
        y = merged_df["is_iceberg"].values

    # Save processed data to cache
    np.save(cache_X, X)
    np.save(cache_angle, angles)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, angles, y, ids


def get_loaders(load_cached_data=True):
    """
    Orchestrates data loading, processing, and DataLoader creation.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Calculate global median angle from training metadata for consistent imputation
    angle_median = get_median_angle()

    # 2. Load/Process Data Splits
    # Train Data
    X_train, ang_train, y_train, ids_train = process_data(
        Config.TRAIN_META_PATH,
        Config.TRAIN_JSON,
        "train",
        angle_median,
        load_cached_data,
    )

    # Validation Data
    X_val, ang_val, y_val, ids_val = process_data(
        Config.VAL_META_PATH, Config.TRAIN_JSON, "val", angle_median, load_cached_data
    )

    # Test Data
    X_test, ang_test, y_test, ids_test = process_data(
        Config.TEST_META_PATH, Config.TEST_JSON, "test", angle_median, load_cached_data
    )

    # 3. Handle Debug Mode (Subsampling)
    if Config.DEBUG:
        limit = Config.MAX_DEBUG_SAMPLES
        print(f"Debug Mode: Limiting datasets to {limit} samples.")
        X_train, ang_train, y_train, ids_train = (
            X_train[:limit],
            ang_train[:limit],
            y_train[:limit],
            ids_train[:limit],
        )
        X_val, ang_val, y_val, ids_val = (
            X_val[:limit],
            ang_val[:limit],
            y_val[:limit],
            ids_val[:limit],
        )
        X_test, ang_test, ids_test = X_test[:limit], ang_test[:limit], ids_test[:limit]

    # 4. Define Augmentations (Train only)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # 5. Instantiate Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, ids_train, transform=train_transform, mode="train"
    )

    val_dataset = IcebergDataset(
        X_val, ang_val, y_val, ids_val, transform=None, mode="val"
    )

    test_dataset = IcebergDataset(
        X_test, ang_test, None, ids_test, transform=None, mode="test"
    )

    # 6. Create DataLoaders
    # Pin memory enables faster data transfer to CUDA devices
    pin = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader
