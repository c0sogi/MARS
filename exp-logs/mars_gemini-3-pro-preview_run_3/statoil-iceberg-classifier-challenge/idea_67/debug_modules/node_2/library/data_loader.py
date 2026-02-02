import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

import library.config as config
from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    CACHE_DIR,
    SEED,
    IMAGE_SIZE,
)
from library.utils import set_seed


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        PyTorch Dataset for Iceberg/Ship classification.

        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            ids (np.ndarray, optional): Shape (N,)
            transform (callable, optional): Augmentation pipeline
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert to tensor (C, H, W)
        # Images are already float32 from caching step
        image = torch.from_numpy(self.images[idx])
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Apply transforms if present (expects tensor or PIL, here tensor)
        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "angle": angle}

        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def process_json_data(json_path, is_train=True):
    """
    Reads JSON, processes bands into (3, 75, 75) images, handles angles and labels.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Process Images
    # Band 1 and Band 2 are lists of 5625 floats
    band_1 = np.array([np.array(b).reshape(75, 75) for b in df["band_1"]])
    band_2 = np.array([np.array(b).reshape(75, 75) for b in df["band_2"]])

    # Band 3: Average of Band 1 and Band 2
    band_3 = (band_1 + band_2) / 2.0

    # Stack to (N, 3, 75, 75)
    # Using axis 1 for Channel dimension (N, C, H, W)
    images = np.stack([band_1, band_2, band_3], axis=1).astype(np.float32)

    # Process Angles
    # Coerce 'na' to NaN
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    # Process IDs
    ids = df["id"].values

    labels = None
    if is_train:
        labels = df["is_iceberg"].values.astype(np.float32)

    return images, angles, labels, ids


def load_data(load_cached_data=True):
    """
    Loads data from cache or raw JSON files.
    Returns:
        X_train, angles_train, y_train, ids_train, X_test, angles_test, ids_test
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache file paths
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "angles_train": os.path.join(CACHE_DIR, "angle_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "ids_train": os.path.join(CACHE_DIR, "ids_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "angles_test": os.path.join(CACHE_DIR, "angle_test.npy"),
        "ids_test": os.path.join(CACHE_DIR, "ids_test.npy"),
    }

    all_exist = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_exist:
        print("Loading data from cache...")
        X_train = np.load(cache_files["X_train"])
        angles_train = np.load(cache_files["angles_train"])
        y_train = np.load(cache_files["y_train"])
        ids_train = np.load(cache_files["ids_train"], allow_pickle=True)

        X_test = np.load(cache_files["X_test"])
        angles_test = np.load(cache_files["angles_test"])
        ids_test = np.load(cache_files["ids_test"], allow_pickle=True)
    else:
        print("Processing raw data...")
        # Train Data
        X_train, angles_train, y_train, ids_train = process_json_data(
            TRAIN_JSON, is_train=True
        )

        # Test Data
        X_test, angles_test, _, ids_test = process_json_data(TEST_JSON, is_train=False)

        # Save to cache
        print("Saving data to cache...")
        np.save(cache_files["X_train"], X_train)
        np.save(cache_files["angles_train"], angles_train)
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["ids_train"], ids_train)

        np.save(cache_files["X_test"], X_test)
        np.save(cache_files["angles_test"], angles_test)
        np.save(cache_files["ids_test"], ids_test)

    return X_train, angles_train, y_train, ids_train, X_test, angles_test, ids_test


def get_fold_loaders(fold_idx, load_cached_data=True):
    """
    Generates DataLoaders for a specific fold with leak-free angle imputation.

    Args:
        fold_idx (int): Index of the current fold (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        train_loader, val_loader
    """
    set_seed(SEED)

    # Load all data
    X, angles, y, ids, _, _, _ = load_data(load_cached_data)

    # Debug slicing
    if config.DEBUG:
        X = X[: config.MAX_DEBUG_SAMPLES]
        angles = angles[: config.MAX_DEBUG_SAMPLES]
        y = y[: config.MAX_DEBUG_SAMPLES]
        ids = ids[: config.MAX_DEBUG_SAMPLES]

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=SEED)

    # Get indices for the specific fold
    # We iterate to find the correct fold indices
    fold_generator = skf.split(X, y)
    train_idx, val_idx = next(x for i, x in enumerate(fold_generator) if i == fold_idx)

    # Split Data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    ids_train, ids_val = ids[train_idx], ids[val_idx]

    # Angle Imputation (Leak-Free)
    # 1. Extract raw angles
    ang_train_raw = angles[train_idx]
    ang_val_raw = angles[val_idx]

    # 2. Calculate median ONLY on training data
    imputation_value = np.nanmedian(ang_train_raw)

    # 3. Impute missing values in both sets using the training median
    ang_train_filled = ang_train_raw.copy()
    ang_train_filled[np.isnan(ang_train_filled)] = imputation_value

    ang_val_filled = ang_val_raw.copy()
    ang_val_filled[np.isnan(ang_val_filled)] = imputation_value

    # Define Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train_filled, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, ang_val_filled, y_val, ids_val, transform=None)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Generates DataLoader for the test set. Imputes angles using global train median.
    """
    set_seed(SEED)

    # Load data
    X_train, angles_train, _, _, X_test, angles_test, ids_test = load_data(
        load_cached_data
    )

    # Impute Test Angles using Global Train Median
    imputation_value = np.nanmedian(angles_train)

    angles_test_filled = angles_test.copy()
    angles_test_filled[np.isnan(angles_test_filled)] = imputation_value

    # Create Dataset
    test_dataset = IcebergDataset(
        X_test, angles_test_filled, labels=None, ids=ids_test, transform=None
    )

    # Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_loader
