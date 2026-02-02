import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg/Ship classification.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75) float32.
            angles (np.ndarray): Shape (N,) float32.
            labels (np.ndarray, optional): Shape (N,) float32.
            ids (np.ndarray, optional): Shape (N,) string/object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]  # (3, 75, 75)
        angle = self.angles[idx]  # scalar

        # Convert to tensor
        image_tensor = torch.from_numpy(image)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply transforms (augmentation)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Return tuple based on mode (train/val vs test)
        if self.labels is not None:
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return image_tensor, angle_tensor, label_tensor
        else:
            # Test mode requires ID for submission
            id_str = str(self.ids[idx])
            return image_tensor, angle_tensor, id_str


def process_data(load_cached_data=True):
    """
    Loads raw data, processes it (reshaping, channel creation, imputation),
    and caches the result to disk as .npy files.
    """
    # Define cache file paths
    cache_files = {
        "X_train": "X_train.npy",
        "angle_train": "angle_train.npy",
        "y_train": "y_train.npy",
        "X_val": "X_val.npy",
        "angle_val": "angle_val.npy",
        "y_val": "y_val.npy",
        "X_test": "X_test.npy",
        "angle_test": "angle_test.npy",
        "ids_test": "ids_test.npy",
    }

    # Check if all cache files exist
    cache_exists = all(
        os.path.exists(os.path.join(Config.IDEA_DIR, fname))
        for fname in cache_files.values()
    )

    if load_cached_data and cache_exists:
        # print("Loading processed data from cache...")
        data = {}
        for key, fname in cache_files.items():
            path = os.path.join(Config.IDEA_DIR, fname)
            data[key] = np.load(path, allow_pickle=True)
        return data

    print("Processing data from scratch (Cache miss or forced reload)...")

    # Ensure output directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Load Raw JSON Data
    # Note: Reading JSON is memory intensive but fits within available RAM.
    with open(Config.TRAIN_JSON_PATH, "r") as f:
        train_json_raw = json.load(f)

    with open(Config.TEST_JSON_PATH, "r") as f:
        test_json_raw = json.load(f)

    # Helper function to process a subset based on metadata indices
    def process_subset(meta_df, raw_data, is_test=False):
        indices = meta_df["original_index"].values

        # Extract subset from raw list
        subset_raw = [raw_data[i] for i in indices]

        # Extract Band 1 and Band 2
        b1 = np.array([item["band_1"] for item in subset_raw], dtype=np.float32)
        b2 = np.array([item["band_2"] for item in subset_raw], dtype=np.float32)

        # Reshape to (N, 75, 75)
        b1 = b1.reshape(-1, 75, 75)
        b2 = b2.reshape(-1, 75, 75)

        # Create Band 3 (Average of HH and HV)
        b3 = (b1 + b2) / 2.0

        # Stack channels: (N, 3, 75, 75)
        # PyTorch expects (C, H, W), so we stack on axis 1
        X = np.stack([b1, b2, b3], axis=1)

        # Extract Angles (with NaNs from metadata)
        angles = meta_df["inc_angle"].values.astype(np.float32)

        # Extract Labels or IDs
        if not is_test:
            y = meta_df["is_iceberg"].values.astype(np.float32)
            ids = None
        else:
            y = None
            ids = meta_df["id"].values

        return X, angles, y, ids

    # Process splits
    X_train, angle_train, y_train, _ = process_subset(train_meta, train_json_raw)
    X_val, angle_val, y_val, _ = process_subset(val_meta, train_json_raw)
    X_test, angle_test, _, ids_test = process_subset(
        test_meta, test_json_raw, is_test=True
    )

    # Impute Missing Incidence Angles
    # Compute median from TRAIN set only to prevent leakage
    angle_median = np.nanmedian(angle_train)

    # Fill NaNs in all sets
    angle_train[np.isnan(angle_train)] = angle_median
    angle_val[np.isnan(angle_val)] = angle_median
    angle_test[np.isnan(angle_test)] = angle_median

    # Prepare data dictionary
    data = {
        "X_train": X_train,
        "angle_train": angle_train,
        "y_train": y_train,
        "X_val": X_val,
        "angle_val": angle_val,
        "y_val": y_val,
        "X_test": X_test,
        "angle_test": angle_test,
        "ids_test": ids_test,
    }

    # Save to cache
    for key, fname in cache_files.items():
        path = os.path.join(Config.IDEA_DIR, fname)
        np.save(path, data[key])

    return data


def get_loaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load data (cached or fresh)
    data = process_data(load_cached_data=load_cached_data)

    X_train, angle_train, y_train = (
        data["X_train"],
        data["angle_train"],
        data["y_train"],
    )
    X_val, angle_val, y_val = data["X_val"], data["angle_val"], data["y_val"]
    X_test, angle_test, ids_test = data["X_test"], data["angle_test"], data["ids_test"]

    # Debugging: Truncate data if needed
    if debug:
        subset_size = Config.DEBUG_SUBSET_SIZE
        X_train = X_train[:subset_size]
        angle_train = angle_train[:subset_size]
        y_train = y_train[:subset_size]

        X_val = X_val[:subset_size]
        angle_val = angle_val[:subset_size]
        y_val = y_val[:subset_size]

        X_test = X_test[:subset_size]
        angle_test = angle_test[:subset_size]
        ids_test = ids_test[:subset_size]

    # Define Transforms
    # Random flips for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # No TTA for val/test
    val_transform = None
    test_transform = None

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angle_train, labels=y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val, angle_val, labels=y_val, transform=val_transform
    )
    test_dataset = IcebergDataset(
        X_test, angle_test, ids=ids_test, transform=test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
