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


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Image data of shape (N, 75, 75, 3).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): Image IDs of shape (N,).
            transform (callable, optional): Transformations to apply to the images.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve raw data
        img = self.X[idx]  # Shape: (75, 75, 3)
        angle = self.angles[idx]

        # Convert to Tensor and rearrange to (C, H, W)
        # Input is float32 (dB values), so we keep it as float
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)

        # Apply augmentations if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)

        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            # For test set, return ID to map predictions
            id_val = self.ids[idx]
            return img_tensor, angle_tensor, id_val


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes images and angles, and caches them as .npy files.
    Implements the logic to generate the 3rd band (average) and impute missing angles.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (X_train, y_train, angle_train, X_test, id_test, angle_test)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    paths = {
        "X_train": Config.CACHE_X_TRAIN,
        "y_train": Config.CACHE_Y_TRAIN,
        "angle_train": Config.CACHE_ANGLE_TRAIN,
        "X_test": Config.CACHE_X_TEST,
        "id_test": Config.CACHE_ID_TEST,
        "angle_test": Config.CACHE_ANGLE_TEST,
    }

    # Check if cache exists
    all_exist = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and all_exist:
        X_train = np.load(paths["X_train"])
        y_train = np.load(paths["y_train"])
        angle_train = np.load(paths["angle_train"])
        X_test = np.load(paths["X_test"])
        id_test = np.load(paths["id_test"])
        angle_test = np.load(paths["angle_test"])
        return X_train, y_train, angle_train, X_test, id_test, angle_test

    # --- Process Training Data ---
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    df_train = pd.DataFrame(train_data)

    def process_bands(df):
        # Reshape flattened bands to 75x75
        b1 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
        )
        b2 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
        )
        # 3rd Channel: Average of HH and HV
        b3 = (b1 + b2) / 2.0
        # Stack to (N, 75, 75, 3)
        return np.stack([b1, b2, b3], axis=-1)

    X_train = process_bands(df_train)
    y_train = df_train["is_iceberg"].values.astype(np.float32)

    # Handle Incidence Angle
    # Convert 'na' to NaN, then calculate median from training set
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    angle_median = df_train["inc_angle"].median()
    df_train["inc_angle"] = df_train["inc_angle"].fillna(angle_median)
    angle_train = df_train["inc_angle"].values.astype(np.float32)

    # --- Process Test Data ---
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)
    df_test = pd.DataFrame(test_data)

    X_test = process_bands(df_test)
    id_test = df_test["id"].values

    # Impute Test Angles using TRAINING median
    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
    df_test["inc_angle"] = df_test["inc_angle"].fillna(angle_median)
    angle_test = df_test["inc_angle"].values.astype(np.float32)

    # --- Save to Cache ---
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["angle_train"], angle_train)
    np.save(paths["X_test"], X_test)
    np.save(paths["id_test"], id_test)
    np.save(paths["angle_test"], angle_test)

    return X_train, y_train, angle_train, X_test, id_test, angle_test


def get_cv_loaders(fold_idx, load_cached_data=True, max_samples=None):
    """
    Creates Train and Validation DataLoaders for a specific fold using Stratified K-Fold.

    Args:
        fold_idx (int): Index of the fold (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached data.
        max_samples (int, optional): Limit dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load full training data
    X, y, angles, _, _, _ = process_and_cache_data(load_cached_data)

    # Debugging: Limit data size
    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]
        angles = angles[:max_samples]

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Find indices for the current fold
    # skf.split returns a generator
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(skf.split(X, y)):
        if i == fold_idx:
            train_idx, val_idx = t_idx, v_idx
            break

    if train_idx is None:
        raise ValueError(f"Fold {fold_idx} out of bounds for {Config.N_FOLDS} splits.")

    # Slice data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angle_train, angle_val = angles[train_idx], angles[val_idx]

    # Define Transforms
    # Training: Random Flips
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Validation: No augmentation
    val_transform = None

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angle_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angle_val, y_val, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,  # Drop incomplete batches for BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates the Test DataLoader.

    Returns:
        DataLoader: DataLoader for the test set.
    """
    _, _, _, X_test, id_test, angle_test = process_and_cache_data(load_cached_data)

    test_dataset = IcebergDataset(X_test, angle_test, ids=id_test, transform=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader
