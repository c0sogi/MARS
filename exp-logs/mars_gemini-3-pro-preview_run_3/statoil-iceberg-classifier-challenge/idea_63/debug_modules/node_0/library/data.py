import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    WORKING_DIR,
    SEED,
    NUM_WORKERS,
    IMAGE_SIZE,
    DEBUG,
    DEBUG_SAMPLES,
)
from library.utils import set_seed

# Ensure reproducible splits
set_seed(SEED)


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert numpy array to tensor
        # X is (3, 75, 75), float32
        image = torch.from_numpy(self.X[idx]).float()
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return image, angle, label
        else:
            return image, angle


def process_json_data(json_path, is_train=True):
    """
    Reads JSON, processes bands into images, extracts angles and labels.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if DEBUG:
        data = data[:DEBUG_SAMPLES]

    df = pd.DataFrame(data)

    # Process Images
    # Band 1 and Band 2 are lists of floats.
    # We need to reshape them to (75, 75)

    images = []
    for i, row in df.iterrows():
        b1 = np.array(row["band_1"]).reshape(75, 75)
        b2 = np.array(row["band_2"]).reshape(75, 75)
        avg = (b1 + b2) / 2.0

        # Stack to (3, 75, 75)
        img = np.stack([b1, b2, avg], axis=0)
        images.append(img)

    X = np.array(images, dtype=np.float32)

    # Process Angles
    # Replace 'na' with NaN and convert to float
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    # Process Labels and IDs
    if is_train:
        y = df["is_iceberg"].values.astype(np.float32)
        ids = df["id"].values
        return X, angles, y, ids
    else:
        ids = df["id"].values
        return X, angles, ids


def load_data(load_cached_data=True):
    """
    Loads data from cache or raw JSON files.
    Returns:
        X_train, angles_train, y_train, ids_train
        X_test, angles_test, ids_test
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train.npy"),
        "angles_train": os.path.join(WORKING_DIR, "angles_train.npy"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "ids_train": os.path.join(WORKING_DIR, "ids_train.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.npy"),
        "angles_test": os.path.join(WORKING_DIR, "angles_test.npy"),
        "ids_test": os.path.join(WORKING_DIR, "ids_test.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = np.load(cache_files["X_train"])
        angles_train = np.load(cache_files["angles_train"])
        y_train = np.load(cache_files["y_train"])
        ids_train = np.load(cache_files["ids_train"])

        X_test = np.load(cache_files["X_test"])
        angles_test = np.load(cache_files["angles_test"])
        ids_test = np.load(cache_files["ids_test"])

    else:
        print("Processing raw data...")
        X_train, angles_train, y_train, ids_train = process_json_data(
            TRAIN_JSON, is_train=True
        )
        X_test, angles_test, ids_test = process_json_data(TEST_JSON, is_train=False)

        # Save to cache
        np.save(cache_files["X_train"], X_train)
        np.save(cache_files["angles_train"], angles_train)
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["ids_train"], ids_train)

        np.save(cache_files["X_test"], X_test)
        np.save(cache_files["angles_test"], angles_test)
        np.save(cache_files["ids_test"], ids_test)

    return (X_train, angles_train, y_train, ids_train), (X_test, angles_test, ids_test)


def get_loaders(fold_idx, batch_size, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold in 5-fold CV.
    Handles leak-free angle imputation.
    """
    (X_train_full, angles_train_full, y_train_full, _), _ = load_data(load_cached_data)

    # Define Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Get indices for the specific fold
    # skf.split requires X and y, but X can be zeros as long as length matches
    splits = list(skf.split(np.zeros(len(y_train_full)), y_train_full))
    train_idx, val_idx = splits[fold_idx]

    # Split data
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
    angles_train, angles_val = angles_train_full[train_idx], angles_train_full[val_idx]

    # Leak-Free Angle Imputation
    # Calculate median on training set ONLY
    valid_angles_train = angles_train[~np.isnan(angles_train)]
    if len(valid_angles_train) > 0:
        median_angle = np.median(valid_angles_train)
    else:
        median_angle = 0.0  # Fallback

    # Impute
    angles_train = np.where(np.isnan(angles_train), median_angle, angles_train)
    angles_val = np.where(np.isnan(angles_val), median_angle, angles_val)

    # Define Transforms
    # Random Horizontal and Vertical Flip for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # No transform for validation
    val_transform = None

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angles_val, y_val, transform=val_transform)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(batch_size, load_cached_data=True):
    """
    Creates DataLoader for the test set.
    Imputes angles using median from the FULL training set.
    """
    (X_train_full, angles_train_full, _, _), (X_test, angles_test, ids_test) = (
        load_data(load_cached_data)
    )

    # Calculate median from full training set
    valid_angles_train = angles_train_full[~np.isnan(angles_train_full)]
    if len(valid_angles_train) > 0:
        median_angle = np.median(valid_angles_train)
    else:
        median_angle = 0.0

    # Impute test angles
    angles_test = np.where(np.isnan(angles_test), median_angle, angles_test)

    test_dataset = IcebergDataset(X_test, angles_test, y=None, transform=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, ids_test
