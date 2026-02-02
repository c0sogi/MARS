import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import seed_everything


class IcebergDataset(Dataset):
    """
    Custom Dataset for Ship vs. Iceberg classification.
    Constructs 3-channel images (HH, HV, Avg) and handles incidence angles.
    """

    def __init__(self, X, angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Image data of shape (N, 75, 75, 3).
            angles (np.ndarray): Normalized incidence angles of shape (N,).
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
        img = self.X[idx]  # Shape: (75, 75, 3)
        angle = self.angles[idx]

        # Convert to tensor
        # Input data is float32 (dB values), not uint8
        img_tensor = torch.from_numpy(img).float()

        # Permute dimensions to (C, H, W) -> (3, 75, 75) for PyTorch
        img_tensor = img_tensor.permute(2, 0, 1)

        # Apply transforms (augmentation) if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Convert angle to tensor
        angle_tensor = torch.tensor([angle], dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor([self.y[idx]], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


def load_and_process_data(load_cached_data=True):
    """
    Loads data from JSON files or Cache, processes bands, handles angles,
    and returns numpy arrays for train, val, and test sets.

    Implements:
    - 3-channel stacking (Band 1, Band 2, Average)
    - Incidence angle imputation (Training Median)
    - Incidence angle normalization (StandardScaler fit on Training)
    - Caching mechanism
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(Config.CACHE_TRAIN_X)
            and os.path.exists(Config.CACHE_TRAIN_Y)
            and os.path.exists(Config.CACHE_TRAIN_ANGLE)
            and os.path.exists(Config.CACHE_VAL_X)
            and os.path.exists(Config.CACHE_VAL_Y)
            and os.path.exists(Config.CACHE_VAL_ANGLE)
            and os.path.exists(Config.CACHE_TEST_X)
            and os.path.exists(Config.CACHE_TEST_IDS)
            and os.path.exists(Config.CACHE_TEST_ANGLE)
        ):

            print("Loading data from cache...")
            X_train = np.load(Config.CACHE_TRAIN_X)
            y_train = np.load(Config.CACHE_TRAIN_Y)
            angle_train = np.load(Config.CACHE_TRAIN_ANGLE)

            X_val = np.load(Config.CACHE_VAL_X)
            y_val = np.load(Config.CACHE_VAL_Y)
            angle_val = np.load(Config.CACHE_VAL_ANGLE)

            X_test = np.load(Config.CACHE_TEST_X)
            ids_test = np.load(Config.CACHE_TEST_IDS)
            angle_test = np.load(Config.CACHE_TEST_ANGLE)

            return (
                (X_train, y_train, angle_train),
                (X_val, y_val, angle_val),
                (X_test, ids_test, angle_test),
            )

    print("Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # 3. Load Raw JSON Data
    # train.json contains both train and val samples
    print(f"Loading {Config.TRAIN_JSON}...")
    with open(Config.TRAIN_JSON, "r") as f:
        train_json_data = json.load(f)
    train_json_df = pd.DataFrame(train_json_data)

    print(f"Loading {Config.TEST_JSON}...")
    with open(Config.TEST_JSON, "r") as f:
        test_json_data = json.load(f)
    test_json_df = pd.DataFrame(test_json_data)

    # 4. Processing Helper Function
    def process_subset(meta_df, raw_df, is_test=False):
        # Merge metadata with raw data on 'id'
        # meta_df contains the cleaned 'inc_angle' (numeric/NaN)
        # raw_df contains 'band_1', 'band_2'
        merged = pd.merge(meta_df, raw_df, on="id", how="left", suffixes=("", "_raw"))

        # Process Images
        # Band 1 (HH)
        b1 = np.array(
            [
                np.array(band)
                .astype(np.float32)
                .reshape(Config.IMAGE_SIZE, Config.IMAGE_SIZE)
                for band in merged["band_1"]
            ]
        )
        # Band 2 (HV)
        b2 = np.array(
            [
                np.array(band)
                .astype(np.float32)
                .reshape(Config.IMAGE_SIZE, Config.IMAGE_SIZE)
                for band in merged["band_2"]
            ]
        )
        # Band 3 (Average)
        b3 = (b1 + b2) / 2.0

        # Stack to (N, 75, 75, 3)
        X = np.stack([b1, b2, b3], axis=-1)

        # Extract Angles (from metadata column, which handles 'na' as NaN)
        angles = merged["inc_angle"].values.astype(np.float32)

        if is_test:
            ids = merged["id"].values
            return X, angles, ids
        else:
            y = merged["is_iceberg"].values.astype(np.float32)
            return X, angles, y

    # Process splits
    print("Processing Train split...")
    X_train, angle_train_raw, y_train = process_subset(train_meta, train_json_df)

    print("Processing Val split...")
    X_val, angle_val_raw, y_val = process_subset(val_meta, train_json_df)

    print("Processing Test split...")
    X_test, angle_test_raw, ids_test = process_subset(
        test_meta, test_json_df, is_test=True
    )

    # 5. Impute Missing Angles
    # Calculate median from valid training angles
    valid_mask = ~np.isnan(angle_train_raw)
    median_angle = np.median(angle_train_raw[valid_mask])

    # Fill NaNs
    angle_train_filled = np.where(
        np.isnan(angle_train_raw), median_angle, angle_train_raw
    )
    angle_val_filled = np.where(np.isnan(angle_val_raw), median_angle, angle_val_raw)
    angle_test_filled = np.where(np.isnan(angle_test_raw), median_angle, angle_test_raw)

    # 6. Normalize Angles
    # Fit StandardScaler on TRAINING data only
    scaler = StandardScaler()
    scaler.fit(angle_train_filled.reshape(-1, 1))

    angle_train = scaler.transform(angle_train_filled.reshape(-1, 1)).flatten()
    angle_val = scaler.transform(angle_val_filled.reshape(-1, 1)).flatten()
    angle_test = scaler.transform(angle_test_filled.reshape(-1, 1)).flatten()

    # 7. Save to Cache
    print("Saving processed data to cache...")
    np.save(Config.CACHE_TRAIN_X, X_train)
    np.save(Config.CACHE_TRAIN_Y, y_train)
    np.save(Config.CACHE_TRAIN_ANGLE, angle_train)

    np.save(Config.CACHE_VAL_X, X_val)
    np.save(Config.CACHE_VAL_Y, y_val)
    np.save(Config.CACHE_VAL_ANGLE, angle_val)

    np.save(Config.CACHE_TEST_X, X_test)
    np.save(Config.CACHE_TEST_IDS, ids_test)
    np.save(Config.CACHE_TEST_ANGLE, angle_test)

    return (
        (X_train, y_train, angle_train),
        (X_val, y_val, angle_val),
        (X_test, ids_test, angle_test),
    )


def get_data_loaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, truncates datasets for debugging.

    Returns:
        train_loader, val_loader, test_loader, ids_test
    """
    # Load data
    (
        (X_train, y_train, angle_train),
        (X_val, y_val, angle_val),
        (X_test, ids_test, angle_test),
    ) = load_and_process_data(load_cached_data)

    # Debug mode: truncate data
    if debug or Config.DEBUG:
        print(f"Debug mode enabled. Truncating data to {Config.DEBUG_SAMPLES} samples.")
        limit = Config.DEBUG_SAMPLES
        X_train, y_train, angle_train = (
            X_train[:limit],
            y_train[:limit],
            angle_train[:limit],
        )
        X_val, y_val, angle_val = X_val[:limit], y_val[:limit], angle_val[:limit]
        X_test, ids_test, angle_test = (
            X_test[:limit],
            ids_test[:limit],
            angle_test[:limit],
        )

    # Define Transforms
    # Augmentation for training: Horizontal and Vertical Flips
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Instantiate Datasets
    train_dataset = IcebergDataset(
        X_train, angle_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angle_val, y_val, transform=None)
    test_dataset = IcebergDataset(X_test, angle_test, y=None, transform=None)

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, ids_test
