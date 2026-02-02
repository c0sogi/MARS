import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library import config, utils

# Logger
logger = utils.get_logger("data_loader")


def load_json_data(file_path):
    """Loads JSON data from a file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def process_data(load_cached_data=True):
    """
    Loads raw data, processes it into numpy arrays, computes global scaling stats,
    and caches the result.

    Returns:
        dict: A dictionary containing processed numpy arrays and scaling stats.
    """
    cache_file = os.path.join(config.CACHE_DIR, "processed_data.npz")

    # 1. Load from Cache if available
    if load_cached_data and os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}")
        try:
            # allow_pickle=True is required to load the dictionary-like structure correctly
            # although we are saving standard arrays, np.savez creates a structure that sometimes triggers this safety check.
            data = np.load(cache_file, allow_pickle=True)
            return {k: data[k] for k in data.files}
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reprocessing...")

    logger.info("Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "test.csv"))

    # 3. Load Raw JSON
    # train.json contains the source data for both train and val splits
    train_json_path = os.path.join(config.INPUT_DIR, "train.json")
    test_json_path = os.path.join(config.INPUT_DIR, "test.json")

    raw_train_data = load_json_data(train_json_path)
    raw_test_data = load_json_data(test_json_path)

    # Create ID lookup maps
    train_data_map = {item["id"]: item for item in raw_train_data}
    test_data_map = {item["id"]: item for item in raw_test_data}

    # 4. Feature Extraction Helper
    def extract_features(meta_df, data_map, is_test=False):
        ids = meta_df["id"].values
        X = []
        inc_angles = []
        y = []

        for img_id in ids:
            item = data_map[img_id]

            # Extract Bands and Reshape
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            # Construct 3rd Channel: Mean of B1 and B2
            b3 = (b1 + b2) / 2.0

            # Stack channels -> (3, 75, 75)
            img = np.stack([b1, b2, b3], axis=0)
            X.append(img)

            # Extract Incidence Angle
            # We use the value from metadata (which might be NaN if original was 'na')
            # However, for consistency with the raw data extraction flow, we can also look at the JSON.
            # The metadata CSV has cleaned numeric/NaN values, so we use that.
            inc = meta_df.loc[meta_df["id"] == img_id, "inc_angle"].values[0]
            inc_angles.append(inc)

            if not is_test:
                y.append(item["is_iceberg"])

        X = np.array(X, dtype=np.float32)
        inc_angles = np.array(inc_angles, dtype=np.float32)
        if not is_test:
            y = np.array(y, dtype=np.float32)
        else:
            y = None

        return X, inc_angles, y

    logger.info("Extracting Train features...")
    X_train, inc_train, y_train = extract_features(train_meta, train_data_map)

    logger.info("Extracting Val features...")
    X_val, inc_val, y_val = extract_features(val_meta, train_data_map)

    logger.info("Extracting Test features...")
    X_test, inc_test, _ = extract_features(test_meta, test_data_map, is_test=True)

    # 5. Handle Incidence Angle Missing Values
    # Impute with mean of the TRAINING set only to prevent leakage
    valid_inc_train = inc_train[~np.isnan(inc_train)]
    inc_mean = np.mean(valid_inc_train)

    inc_train = np.nan_to_num(inc_train, nan=inc_mean)
    inc_val = np.nan_to_num(inc_val, nan=inc_mean)
    inc_test = np.nan_to_num(inc_test, nan=inc_mean)

    # 6. Compute Global Scaling Statistics
    # Use entire labeled dataset (Train + Val) for robust global min/max
    X_full_train = np.concatenate([X_train, X_val], axis=0)

    # Compute min and max per channel (Shape: 3)
    # Axis 0=Batch, 2=H, 3=W. We want stats across Batch, H, and W for each Channel (Axis 1)
    # Wait, in extract_features we stacked as (3, 75, 75). So shape is (N, 3, 75, 75).
    # We want to reduce over axes (0, 2, 3).
    ch_mins = np.min(X_full_train, axis=(0, 2, 3))
    ch_maxs = np.max(X_full_train, axis=(0, 2, 3))

    logger.info(f"Global Stats - Mins: {ch_mins}, Maxs: {ch_maxs}")

    # 7. Cache Results
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez(
        cache_file,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_val=X_val,
        y_val=y_val,
        inc_val=inc_val,
        X_test=X_test,
        inc_test=inc_test,
        ch_mins=ch_mins,
        ch_maxs=ch_maxs,
    )

    logger.info("Data processing complete and cached.")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "inc_train": inc_train,
        "X_val": X_val,
        "y_val": y_val,
        "inc_val": inc_val,
        "X_test": X_test,
        "inc_test": inc_test,
        "ch_mins": ch_mins,
        "ch_maxs": ch_maxs,
    }


class IcebergDataset(Dataset):
    def __init__(self, X, inc_angles, labels=None, transform=None, scaling_stats=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75)
            inc_angles (np.ndarray): Incidence angles of shape (N,)
            labels (np.ndarray, optional): Labels of shape (N,)
            transform (callable, optional): Augmentation function
            scaling_stats (tuple, optional): (mins, maxs) for global normalization
        """
        self.X = X
        self.inc_angles = inc_angles
        self.labels = labels
        self.transform = transform
        self.scaling_stats = scaling_stats

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Copy to avoid modifying the original array
        img = self.X[idx].copy()
        inc = self.inc_angles[idx]

        # 1. Global Normalization
        if self.scaling_stats:
            mins, maxs = self.scaling_stats
            # Reshape for broadcasting: (3, 1, 1)
            mins = mins.reshape(3, 1, 1)
            maxs = maxs.reshape(3, 1, 1)

            # Apply Min-Max Scaling
            denom = maxs - mins
            # Avoid division by zero
            denom[denom == 0] = 1.0
            img = (img - mins) / denom

            # NOTE: No clipping is applied, allowing outliers > 1.0 or < 0.0

        # 2. Augmentation
        if self.transform:
            img = self.transform(img)

        # 3. Convert to Tensor
        img_tensor = torch.from_numpy(img).float()
        inc_tensor = torch.tensor([inc], dtype=torch.float32)  # Wrap scalar in tensor

        if self.labels is not None:
            label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
            return img_tensor, inc_tensor, label_tensor
        else:
            return img_tensor, inc_tensor


def get_transforms(augment=False):
    """
    Returns a transformation function.
    """

    def transform_fn(img):
        # img is numpy array (3, 75, 75)
        if augment:
            # Random Rotation (0, 90, 180, 270)
            k = np.random.choice([0, 1, 2, 3])
            img = np.rot90(img, k=k, axes=(1, 2))

            # Random Horizontal Flip
            # Axis 2 is width
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2)

            # Vertical Flip is explicitly excluded

        return img.copy()  # Return copy to ensure positive memory strides for PyTorch

    return transform_fn


def get_loaders(
    batch_size=config.BATCH_SIZE, debug=config.DEBUG, load_cached_data=True
):
    """
    Constructs and returns DataLoaders for train, validation, and test sets.
    """
    # Load processed data
    data = process_data(load_cached_data=load_cached_data)

    X_train = data["X_train"]
    y_train = data["y_train"]
    inc_train = data["inc_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]
    inc_val = data["inc_val"]

    X_test = data["X_test"]
    inc_test = data["inc_test"]

    # Retrieve global scaling stats
    ch_mins = data["ch_mins"]
    ch_maxs = data["ch_maxs"]
    scaling_stats = (ch_mins, ch_maxs)

    # Debug Mode: Slice datasets
    if debug:
        logger.info(f"Debug mode: trimming datasets to {config.DEBUG_SIZE} samples")
        limit = min(config.DEBUG_SIZE, len(X_train))
        X_train = X_train[:limit]
        y_train = y_train[:limit]
        inc_train = inc_train[:limit]

        limit_val = min(config.DEBUG_SIZE, len(X_val))
        X_val = X_val[:limit_val]
        y_val = y_val[:limit_val]
        inc_val = inc_val[:limit_val]

        limit_test = min(config.DEBUG_SIZE, len(X_test))
        X_test = X_test[:limit_test]
        inc_test = inc_test[:limit_test]

    # Initialize Datasets
    # Train: Augmentation Enabled
    train_ds = IcebergDataset(
        X_train,
        inc_train,
        y_train,
        transform=get_transforms(augment=True),
        scaling_stats=scaling_stats,
    )

    # Val: No Augmentation
    val_ds = IcebergDataset(
        X_val,
        inc_val,
        y_val,
        transform=get_transforms(augment=False),
        scaling_stats=scaling_stats,
    )

    # Test: No Augmentation, No Labels
    test_ds = IcebergDataset(
        X_test,
        inc_test,
        labels=None,
        transform=get_transforms(augment=False),
        scaling_stats=scaling_stats,
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
