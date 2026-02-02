import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data")


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles 3-channel SAR images and incidence angle metadata.
    """

    def __init__(self, X, inc_angles, y=None, transform=False):
        """
        Args:
            X (np.ndarray): Images of shape (N, 75, 75, 3).
            inc_angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (bool): Whether to apply data augmentation.
        """
        self.X = X
        self.inc_angles = inc_angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx].astype(np.float32)  # (75, 75, 3)
        inc = self.inc_angles[idx].astype(np.float32)

        # Apply Augmentation (Training only)
        if self.transform:
            # Random Rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            img = np.rot90(img, k, axes=(0, 1))

            # Random Horizontal Flip
            if np.random.random() < 0.5:
                img = np.fliplr(img)

            # Note: Vertical Flip and Mixup are explicitly excluded per instructions.

        # Convert to Tensor and Channel-First format: (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        x_tensor = torch.from_numpy(img)
        inc_tensor = torch.tensor(inc)

        if self.y is not None:
            y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_tensor, inc_tensor, y_tensor
        else:
            return x_tensor, inc_tensor


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw data, processes it (3-channel construction, normalization, imputation),
    and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        dict: Dictionary containing processed numpy arrays for train, val, and test splits.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "processed_data.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path)
            return dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-processing data.")

    logger.info("Processing data from scratch...")

    # 2. Load Metadata
    logger.info("Loading metadata CSVs...")
    df_train_meta = pd.read_csv(Config.TRAIN_CSV)
    df_val_meta = pd.read_csv(Config.VAL_CSV)
    df_test_meta = pd.read_csv(Config.TEST_CSV)

    # 3. Load Raw JSON Data
    logger.info("Loading raw JSON files (this may take a moment)...")
    with open(Config.TRAIN_JSON, "r") as f:
        train_json_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_json_data = json.load(f)

    # Map ID to Band Data for fast lookup
    # Combine both json sources into one lookup dict
    id_to_bands = {}
    for item in train_json_data:
        id_to_bands[item["id"]] = (item["band_1"], item["band_2"])
    for item in test_json_data:
        id_to_bands[item["id"]] = (item["band_1"], item["band_2"])

    # Helper function to construct image array from metadata df
    def build_dataset(df):
        ids = df["id"].values
        count = len(ids)

        # Pre-allocate arrays
        # Shape: (N, 75, 75, 3)
        X = np.zeros((count, 75, 75, 3), dtype=np.float32)

        for i, img_id in enumerate(ids):
            band1, band2 = id_to_bands[img_id]

            # Reshape to 75x75
            b1 = np.array(band1).reshape(75, 75)
            b2 = np.array(band2).reshape(75, 75)

            # Construct 3rd channel: Average
            avg = (b1 + b2) / 2.0

            # Stack
            X[i, :, :, 0] = b1
            X[i, :, :, 1] = b2
            X[i, :, :, 2] = avg

        return X

    logger.info("Constructing image arrays...")
    X_train = build_dataset(df_train_meta)
    X_val = build_dataset(df_val_meta)
    X_test = build_dataset(df_test_meta)

    # Extract Targets and Incidence Angles from Metadata
    y_train = df_train_meta["is_iceberg"].values.astype(np.float32)
    y_val = df_val_meta["is_iceberg"].values.astype(np.float32)

    inc_train = df_train_meta["inc_angle"].values
    inc_val = df_val_meta["inc_angle"].values
    inc_test = df_test_meta["inc_angle"].values

    # 4. Impute Incidence Angles
    # Calculate mean from training set (ignoring NaNs)
    inc_mean = np.nanmean(inc_train)
    logger.info(f"Imputing missing incidence angles with training mean: {inc_mean:.4f}")

    # Fill NaNs
    inc_train = np.nan_to_num(inc_train, nan=inc_mean).astype(np.float32)
    inc_val = np.nan_to_num(inc_val, nan=inc_mean).astype(np.float32)
    inc_test = np.nan_to_num(inc_test, nan=inc_mean).astype(np.float32)

    # 5. Normalization (Independent Per-Channel Min-Max Scaling)
    logger.info("Applying Per-Channel Min-Max Scaling...")

    # Calculate stats on Training Set ONLY
    # Channels: 0=HH, 1=HV, 2=Avg
    min_vals = np.min(X_train, axis=(0, 1, 2))  # (3,)
    max_vals = np.max(X_train, axis=(0, 1, 2))  # (3,)

    logger.info(f"Channel Mins: {min_vals}")
    logger.info(f"Channel Maxs: {max_vals}")

    # Avoid division by zero
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0

    def normalize(X):
        return (X - min_vals) / denom

    X_train = normalize(X_train)
    X_val = normalize(X_val)
    X_test = normalize(X_test)

    # 6. Save to Cache
    logger.info(f"Saving processed data to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_val=X_val,
        y_val=y_val,
        inc_val=inc_val,
        X_test=X_test,
        inc_test=inc_test,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "inc_train": inc_train,
        "X_val": X_val,
        "y_val": y_val,
        "inc_val": inc_val,
        "X_test": X_test,
        "inc_test": inc_test,
    }


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load data
    data = process_and_cache_data(load_cached_data=load_cached_data)

    X_train = data["X_train"]
    y_train = data["y_train"]
    inc_train = data["inc_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]
    inc_val = data["inc_val"]

    X_test = data["X_test"]
    inc_test = data["inc_test"]

    # Handle Debug Mode
    if Config.DEBUG:
        logger.info(
            f"DEBUG mode active. Subsampling to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        X_train = X_train[: Config.DEBUG_SAMPLE_SIZE]
        y_train = y_train[: Config.DEBUG_SAMPLE_SIZE]
        inc_train = inc_train[: Config.DEBUG_SAMPLE_SIZE]

        X_val = X_val[: Config.DEBUG_SAMPLE_SIZE]
        y_val = y_val[: Config.DEBUG_SAMPLE_SIZE]
        inc_val = inc_val[: Config.DEBUG_SAMPLE_SIZE]

        X_test = X_test[: Config.DEBUG_SAMPLE_SIZE]
        inc_test = inc_test[: Config.DEBUG_SAMPLE_SIZE]

    # Create Datasets
    # Train: Apply Augmentation
    train_dataset = IcebergDataset(X_train, inc_train, y_train, transform=True)

    # Val/Test: No Augmentation
    val_dataset = IcebergDataset(X_val, inc_val, y_val, transform=False)
    test_dataset = IcebergDataset(X_test, inc_test, y=None, transform=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
