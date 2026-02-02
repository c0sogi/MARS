import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library import config


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    """

    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of shape (N, 3, 75, 75).
            angles (np.ndarray): Array of shape (N,).
            labels (np.ndarray, optional): Array of shape (N,).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        img = self.images[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]  # Scalar

        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply augmentations
        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


def _process_json_data(json_path, is_train=True):
    """
    Reads JSON file and converts to numpy arrays.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Pre-allocate lists
    images = []
    angles = []
    ids = []
    labels = [] if is_train else None

    for item in data:
        # Process Images
        # Band 1 and Band 2 are flattened 5625 elements -> 75x75
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)

        # Synthetic 3rd channel: Average of HH and HV
        avg = (b1 + b2) / 2.0

        # Stack to (3, 75, 75)
        img = np.stack([b1, b2, avg], axis=0)
        images.append(img)

        # Process Angle
        ang = item["inc_angle"]
        if ang == "na":
            angles.append(np.nan)
        else:
            angles.append(float(ang))

        # Process ID
        ids.append(item["id"])

        # Process Label
        if is_train:
            labels.append(item["is_iceberg"])

    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    angles = np.array(angles, dtype=np.float32)
    ids = np.array(ids)
    if is_train:
        labels = np.array(labels, dtype=np.float32)

    return images, angles, ids, labels


def load_data(load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from raw JSON.
    Returns tuple of training and test data arrays.
    """
    # Define cache paths
    cache_files = {
        "X_train": os.path.join(config.CACHE_DIR, "X_train.npy"),
        "angle_train": os.path.join(config.CACHE_DIR, "angle_train.npy"),
        "y_train": os.path.join(config.CACHE_DIR, "y_train.npy"),
        "ids_train": os.path.join(config.CACHE_DIR, "ids_train.npy"),
        "X_test": os.path.join(config.CACHE_DIR, "X_test.npy"),
        "angle_test": os.path.join(config.CACHE_DIR, "angle_test.npy"),
        "ids_test": os.path.join(config.CACHE_DIR, "ids_test.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = np.load(cache_files["X_train"])
        angle_train = np.load(cache_files["angle_train"])
        y_train = np.load(cache_files["y_train"])
        ids_train = np.load(cache_files["ids_train"])

        X_test = np.load(cache_files["X_test"])
        angle_test = np.load(cache_files["angle_test"])
        ids_test = np.load(cache_files["ids_test"])

    else:
        print("Processing raw data from JSON...")
        # Ensure cache directory exists
        os.makedirs(config.CACHE_DIR, exist_ok=True)

        # Process Train
        X_train, angle_train, ids_train, y_train = _process_json_data(
            config.TRAIN_JSON, is_train=True
        )

        # Process Test
        X_test, angle_test, ids_test, _ = _process_json_data(
            config.TEST_JSON, is_train=False
        )

        # Save to cache
        print("Saving data to cache...")
        np.save(cache_files["X_train"], X_train)
        np.save(cache_files["angle_train"], angle_train)
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["ids_train"], ids_train)

        np.save(cache_files["X_test"], X_test)
        np.save(cache_files["angle_test"], angle_test)
        np.save(cache_files["ids_test"], ids_test)

    return (X_train, angle_train, y_train, ids_train), (X_test, angle_test, ids_test)


def get_fold_loaders(
    fold_index, total_folds=5, batch_size=32, num_workers=4, load_cached_data=True
):
    """
    Creates DataLoaders for a specific fold with leak-free imputation.
    """
    # Load all data
    (X_all, angle_all, y_all, ids_all), _ = load_data(load_cached_data=load_cached_data)

    # Define Stratified K-Fold
    skf = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=config.SEED)

    # Get indices for the requested fold
    # list(skf.split) returns a list of (train_idx, val_idx) tuples
    splits = list(skf.split(X_all, y_all))
    if fold_index >= total_folds:
        raise ValueError(
            f"Fold index {fold_index} out of range for {total_folds} folds."
        )

    train_idx, val_idx = splits[fold_index]

    # Subset data
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    angle_train = angle_all[
        train_idx
    ].copy()  # Copy to avoid modifying original array during imputation
    angle_val = angle_all[val_idx].copy()

    # Leak-free Imputation
    # Calculate median ONLY on training data
    train_median_angle = np.nanmedian(angle_train)

    # Fill NaNs in training data
    angle_train[np.isnan(angle_train)] = train_median_angle

    # Fill NaNs in validation data using TRAIN median
    angle_val[np.isnan(angle_val)] = train_median_angle

    # Define Transforms
    # Train: Random Flips
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )
    # Val: None
    val_transform = None

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angle_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angle_val, y_val, transform=val_transform)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=32, num_workers=4, load_cached_data=True):
    """
    Creates DataLoader for the test set with imputation based on full training set.
    """
    # Load all data
    (X_train_all, angle_train_all, _, _), (X_test, angle_test, ids_test) = load_data(
        load_cached_data=load_cached_data
    )

    # Imputation using full training set statistics
    full_train_median = np.nanmedian(angle_train_all)

    # Fill NaNs in test data (copy to be safe)
    angle_test_imputed = angle_test.copy()
    angle_test_imputed[np.isnan(angle_test_imputed)] = full_train_median

    # Create Dataset (No transform for test)
    test_dataset = IcebergDataset(
        X_test, angle_test_imputed, labels=None, transform=None
    )

    # Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return test_loader, ids_test
