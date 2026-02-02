import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import set_seed


def load_and_process_data(debug=False, load_cached_data=True):
    """
    Loads and processes the training and test data.
    Implements caching to speed up subsequent runs.

    Args:
        debug (bool): If True, limits the dataset size for debugging.
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (X_train, angles_train, y_train, X_test, angles_test, ids_test)
    """
    # Define cache filenames
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "angles_train": os.path.join(Config.CACHE_DIR, "angles_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "angles_test": os.path.join(Config.CACHE_DIR, "angles_test.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading data from cache at {Config.CACHE_DIR}...")
        X_train = np.load(cache_files["X_train"])
        angles_train = np.load(cache_files["angles_train"])
        y_train = np.load(cache_files["y_train"])
        X_test = np.load(cache_files["X_test"])
        angles_test = np.load(cache_files["angles_test"])
        ids_test = np.load(cache_files["ids_test"])
    else:
        print("Processing data from raw JSON files...")

        # --- Process Training Data ---
        print(f"Loading {Config.TRAIN_JSON}...")
        with open(Config.TRAIN_JSON, "r") as f:
            train_data = json.load(f)

        df_train = pd.DataFrame(train_data)

        # Process Images
        X_train = _process_images(df_train)

        # Process Angles
        # Convert to numeric, coerce errors to NaN
        df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
        # Impute missing values with median
        angle_median = df_train["inc_angle"].median()
        df_train["inc_angle"] = df_train["inc_angle"].fillna(angle_median)
        angles_train = df_train["inc_angle"].values.astype(np.float32)

        # Process Labels
        y_train = df_train["is_iceberg"].values.astype(np.float32)

        # --- Process Test Data ---
        print(f"Loading {Config.TEST_JSON}...")
        with open(Config.TEST_JSON, "r") as f:
            test_data = json.load(f)

        df_test = pd.DataFrame(test_data)

        # Process Images
        X_test = _process_images(df_test)

        # Process Angles
        df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
        # Use TRAIN median for test imputation to avoid leakage/inconsistency
        df_test["inc_angle"] = df_test["inc_angle"].fillna(angle_median)
        angles_test = df_test["inc_angle"].values.astype(np.float32)

        # Process IDs
        ids_test = df_test["id"].values

        # --- Save to Cache ---
        print("Saving processed data to cache...")
        np.save(cache_files["X_train"], X_train)
        np.save(cache_files["angles_train"], angles_train)
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["X_test"], X_test)
        np.save(cache_files["angles_test"], angles_test)
        np.save(cache_files["ids_test"], ids_test)

    # --- Debug Mode ---
    if debug:
        print(f"Debug mode: trimming dataset to {Config.DEBUG_SAMPLES} samples.")
        X_train = X_train[: Config.DEBUG_SAMPLES]
        angles_train = angles_train[: Config.DEBUG_SAMPLES]
        y_train = y_train[: Config.DEBUG_SAMPLES]
        # We also trim test set in debug mode to ensure quick pipeline verification
        X_test = X_test[: Config.DEBUG_SAMPLES]
        angles_test = angles_test[: Config.DEBUG_SAMPLES]
        ids_test = ids_test[: Config.DEBUG_SAMPLES]

    return X_train, angles_train, y_train, X_test, angles_test, ids_test


def _process_images(df):
    """
    Helper to process band data into (N, 3, 75, 75) numpy array.
    Constructs Band 3 as the average of Band 1 and Band 2.
    """
    # Extract bands
    # Each band is a list of 5625 floats. Reshape to 75x75.
    band_1 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
    )
    band_2 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
    )

    # Calculate Band 3 (Average)
    band_3 = (band_1 + band_2) / 2.0

    # Stack channels: (N, 3, 75, 75)
    # np.stack along axis 1 creates the channel dimension
    X = np.stack([band_1, band_2, band_3], axis=1)
    return X


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles both training (with labels) and testing (with IDs).
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None):
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get image and convert to tensor
        img = self.X[idx]  # (3, 75, 75)
        img_tensor = torch.from_numpy(img)

        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Get angle
        angle = self.angles[idx]
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # If training/val (y exists)
        if self.y is not None:
            label = self.y[idx]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor

        # If test (ids exist)
        if self.ids is not None:
            id_val = self.ids[idx]
            return img_tensor, angle_tensor, id_val

        # Fallback
        return img_tensor, angle_tensor


def create_fold_loaders(X, angles, y, fold_idx):
    """
    Creates train and validation DataLoaders for a specific fold using StratifiedKFold.

    Args:
        X (np.ndarray): Training images.
        angles (np.ndarray): Training angles.
        y (np.ndarray): Training labels.
        fold_idx (int): Index of the current fold (0 to NUM_FOLDS-1).

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Generate splits
    # We need to iterate to find the specific fold indices
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(skf.split(X, y)):
        if i == fold_idx:
            train_idx = t_idx
            val_idx = v_idx
            break

    if train_idx is None:
        raise ValueError(f"Invalid fold_idx {fold_idx} for {Config.NUM_FOLDS} folds.")

    # Subset data
    X_train_fold = X[train_idx]
    angles_train_fold = angles[train_idx]
    y_train_fold = y[train_idx]

    X_val_fold = X[val_idx]
    angles_val_fold = angles[val_idx]
    y_val_fold = y[val_idx]

    # Define Transforms
    # RandomHorizontalFlip and RandomVerticalFlip as per Idea
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Create Datasets
    # Train gets augmentation, Val does not
    train_dataset = IcebergDataset(
        X_train_fold, angles_train_fold, y=y_train_fold, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val_fold, angles_val_fold, y=y_val_fold, transform=None
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader


def create_test_loader(X_test, angles_test, ids_test):
    """
    Creates a DataLoader for the test set.
    """
    test_dataset = IcebergDataset(X_test, angles_test, ids=ids_test, transform=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader
