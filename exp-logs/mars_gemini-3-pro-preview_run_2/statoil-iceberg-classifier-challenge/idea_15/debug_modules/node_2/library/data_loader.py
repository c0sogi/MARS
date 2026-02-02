import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library import config, utils


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            transform (bool): Whether to apply data augmentation
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert to tensor
        image = torch.from_numpy(self.images[idx]).float()
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.transform:
            # Random rotation (0, 90, 180, 270 degrees)
            # k is number of times to rotate by 90 degrees
            k = np.random.randint(0, 4)
            image = torch.rot90(image, k, dims=[1, 2])

            # Random horizontal flip (flip along width axis, dim 2)
            if np.random.random() > 0.5:
                image = torch.flip(image, dims=[2])

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            return image, angle


def process_data(load_cached_data=True):
    """
    Loads, preprocesses, and caches the data.

    Returns:
        tuple: (X_train, y_train, inc_train, X_test, inc_test, ids_test)
    """
    if load_cached_data and os.path.exists(config.PROCESSED_DATA_PATH):
        print(f"Loading cached data from {config.PROCESSED_DATA_PATH}")
        data = np.load(config.PROCESSED_DATA_PATH, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["inc_train"],
            data["X_test"],
            data["inc_test"],
            data["ids_test"],
        )

    print("Processing data from scratch...")

    # Load JSON files
    with open(config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # Helper to process images into (N, 3, 75, 75)
    def process_images(df):
        # Band 1 and Band 2 are lists of floats, reshape to 75x75
        b1 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
        )
        b2 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
        )

        # Calculate Average Band
        avg = (b1 + b2) / 2.0

        # Stack channels: (N, 3, 75, 75)
        images = np.stack([b1, b2, avg], axis=1)
        return images

    print("Constructing images...")
    X_train = process_images(df_train)
    X_test = process_images(df_test)

    # Process Incidence Angles
    print("Processing incidence angles...")
    # Coerce to numeric, turning 'na' into NaN
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")

    # Fill NaN with mean of training set to prevent leakage
    inc_mean = df_train["inc_angle"].mean()
    df_train["inc_angle"] = df_train["inc_angle"].fillna(inc_mean)
    df_test["inc_angle"] = df_test["inc_angle"].fillna(inc_mean)

    inc_train = df_train["inc_angle"].values.astype(np.float32)
    inc_test = df_test["inc_angle"].values.astype(np.float32)

    # Process Labels and IDs
    y_train = df_train["is_iceberg"].values.astype(np.float32)
    ids_test = df_test["id"].values

    # Normalization using GlobalMinMaxScaler
    print("Normalizing images...")
    scaler = utils.GlobalMinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Cache results
    print(f"Saving processed data to {config.PROCESSED_DATA_PATH}")
    np.savez(
        config.PROCESSED_DATA_PATH,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_test=X_test,
        inc_test=inc_test,
        ids_test=ids_test,
    )

    return X_train, y_train, inc_train, X_test, inc_test, ids_test


def get_fold_loaders(fold_index, load_cached_data=True):
    """
    Returns train and validation loaders for a specific fold using Stratified K-Fold.

    Args:
        fold_index (int): The index of the fold (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to load data from cache.

    Returns:
        tuple: (train_loader, val_loader)
    """
    X_train_all, y_train_all, inc_train_all, _, _, _ = process_data(load_cached_data)

    if config.DEBUG:
        # Slice for debugging to reduce runtime
        limit = config.MAX_DEBUG_SAMPLES
        X_train_all = X_train_all[:limit]
        y_train_all = y_train_all[:limit]
        inc_train_all = inc_train_all[:limit]
        # Use 2 splits for debug mode to ensure at least one fold works
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=config.SEED)
        if fold_index >= 2:
            fold_index = 0
    else:
        skf = StratifiedKFold(
            n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
        )

    # Generate splits
    # We convert generator to list to access specific fold
    splits = list(skf.split(X_train_all, y_train_all))
    train_idx, val_idx = splits[fold_index]

    # Create datasets
    # Apply transform=True only for training set
    train_dataset = IcebergDataset(
        X_train_all[train_idx],
        inc_train_all[train_idx],
        y_train_all[train_idx],
        transform=True,
    )

    val_dataset = IcebergDataset(
        X_train_all[val_idx],
        inc_train_all[val_idx],
        y_train_all[val_idx],
        transform=False,
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns the test loader and the list of test IDs.

    Returns:
        tuple: (test_loader, ids_test)
    """
    _, _, _, X_test, inc_test, ids_test = process_data(load_cached_data)

    if config.DEBUG:
        limit = config.MAX_DEBUG_SAMPLES
        X_test = X_test[:limit]
        inc_test = inc_test[:limit]
        ids_test = ids_test[:limit]

    test_dataset = IcebergDataset(X_test, inc_test, transform=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, ids_test
