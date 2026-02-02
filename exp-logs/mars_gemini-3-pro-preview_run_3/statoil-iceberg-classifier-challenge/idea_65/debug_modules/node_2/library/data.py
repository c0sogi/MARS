import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    """

    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,). Defaults to None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and convert to tensor
        # images are already float32 from processing
        image = torch.from_numpy(self.images[idx])

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Float for BCE
            return image, angle, label
        else:
            # For test set, return -1 as dummy label or just handle in loop
            # Returning dummy label simplifies the loop structure
            return image, angle, torch.tensor(-1.0)


def _process_json_to_numpy(json_path, is_train=True):
    """
    Reads JSON, processes bands to (N, 3, 75, 75), extracts angles and labels.
    """
    # Using pandas for easier JSON reading
    df = pd.read_json(json_path)

    # Process Images
    # Band 1: HH, Band 2: HV
    # Reshape to (N, 75, 75)
    band_1 = np.array([np.array(b).reshape(75, 75) for b in df["band_1"]]).astype(
        np.float32
    )
    band_2 = np.array([np.array(b).reshape(75, 75) for b in df["band_2"]]).astype(
        np.float32
    )

    # Band 3: Average of Band 1 and Band 2
    band_3 = (band_1 + band_2) / 2.0

    # Stack to (N, 3, 75, 75)
    X = np.stack([band_1, band_2, band_3], axis=1)

    # Process Angles
    # Convert 'na' to NaN and then to numeric
    df["inc_angle"] = pd.to_numeric(df["inc_angle"], errors="coerce")
    angles = df["inc_angle"].values.astype(np.float32)

    # Process IDs
    ids = df["id"].values

    if is_train:
        y = df["is_iceberg"].values.astype(np.float32)
        return X, angles, y, ids
    else:
        return X, angles, None, ids


def _load_data_cached(mode, load_cached_data=True):
    """
    Loads data from cache or processes from raw JSON.
    mode: 'train' or 'test'
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    X_path = os.path.join(cache_dir, f"X_{mode}.npy")
    angle_path = os.path.join(cache_dir, f"angle_{mode}.npy")
    ids_path = os.path.join(cache_dir, f"ids_{mode}.npy")
    y_path = os.path.join(cache_dir, f"y_{mode}.npy")

    files_exist = (
        os.path.exists(X_path)
        and os.path.exists(angle_path)
        and os.path.exists(ids_path)
    )
    if mode == "train":
        files_exist = files_exist and os.path.exists(y_path)

    if load_cached_data and files_exist:
        # Load from cache
        X = np.load(X_path)
        angles = np.load(angle_path)
        ids = np.load(ids_path)
        y = np.load(y_path) if mode == "train" else None
        return X, angles, y, ids
    else:
        # Process from scratch
        json_path = Config.TRAIN_JSON if mode == "train" else Config.TEST_JSON
        X, angles, y, ids = _process_json_to_numpy(
            json_path, is_train=(mode == "train")
        )

        # Save to cache
        np.save(X_path, X)
        np.save(angle_path, angles)
        np.save(ids_path, ids)
        if y is not None:
            np.save(y_path, y)

        return X, angles, y, ids


def get_data_loaders(fold, batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Returns train and validation DataLoaders for a specific fold with leak-free imputation.
    """
    # 1. Load all training data
    X, angles, y, ids = _load_data_cached("train", load_cached_data)

    # 2. Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    # skf.split requires X and y (y for stratification)
    # We iterate to find the specific fold
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(skf.split(X, y)):
        if i == fold:
            train_idx = t_idx
            val_idx = v_idx
            break

    if train_idx is None:
        raise ValueError(f"Fold {fold} out of range for {Config.NUM_FOLDS} folds.")

    # 3. Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angles_train, angles_val = angles[train_idx], angles[val_idx]

    # 4. Leak-Free Imputation
    # Calculate median ONLY on training data
    train_median_angle = np.nanmedian(angles_train)

    # Fill NaNs in training data
    angles_train = np.where(np.isnan(angles_train), train_median_angle, angles_train)

    # Fill NaNs in validation data using TRAIN median
    angles_val = np.where(np.isnan(angles_val), train_median_angle, angles_val)

    # 5. Define Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )
    # No TTA or transforms for validation as per Idea
    val_transform = None

    # 6. Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angles_val, y_val, transform=val_transform)

    # 7. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Returns the test DataLoader. Imputes missing angles using global training median.
    """
    # 1. Load Test Data
    X_test, angles_test, _, ids_test = _load_data_cached("test", load_cached_data)

    # 2. Imputation
    # We need the global training median to impute test values properly
    _, angles_train_all, _, _ = _load_data_cached("train", load_cached_data)
    global_median_angle = np.nanmedian(angles_train_all)

    angles_test = np.where(np.isnan(angles_test), global_median_angle, angles_test)

    # 3. Create Dataset
    test_dataset = IcebergDataset(X_test, angles_test, labels=None, transform=None)

    # 4. Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, ids_test
