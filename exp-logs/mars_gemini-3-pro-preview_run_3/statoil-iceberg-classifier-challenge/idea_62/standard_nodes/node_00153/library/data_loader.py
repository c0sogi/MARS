import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from library.config import Config


def process_json_to_numpy(json_path, is_train=True):
    """
    Reads the raw JSON file and converts it to numpy arrays.
    Constructs 3-channel images: [HH, HV, Avg].
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Process Images
    # Band 1 (HH) and Band 2 (HV) are flattened 75x75 images
    b1 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
    )
    b2 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
    )

    # Band 3 (Average)
    b3 = (b1 + b2) / 2.0

    # Stack into (N, 3, 75, 75)
    X = np.stack([b1, b2, b3], axis=1)

    # Process Angles
    # Convert 'na' to NaN and ensure float
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    # Process IDs
    ids = df["id"].values

    if is_train:
        y = df["is_iceberg"].values.astype(np.float32)
        return X, angles, y, ids
    else:
        return X, angles, ids


def load_dataset_with_cache(mode="train", load_cached_data=True):
    """
    Loads dataset from cache if available, otherwise processes from raw JSON.

    Args:
        mode (str): 'train' or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        Tuple of numpy arrays depending on mode.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_prefix = "train" if mode == "train" else "test"
    path_X = os.path.join(Config.CACHE_DIR, f"X_{cache_prefix}.npy")
    path_angles = os.path.join(Config.CACHE_DIR, f"angle_{cache_prefix}.npy")
    path_ids = os.path.join(Config.CACHE_DIR, f"ids_{cache_prefix}.npy")
    path_y = os.path.join(Config.CACHE_DIR, f"y_{cache_prefix}.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(path_X)
        and os.path.exists(path_angles)
        and os.path.exists(path_ids)
    )
    if mode == "train":
        cache_exists = cache_exists and os.path.exists(path_y)

    if load_cached_data and cache_exists:
        # Load from cache
        X = np.load(path_X)
        angles = np.load(path_angles)
        ids = np.load(path_ids, allow_pickle=True)
        if mode == "train":
            y = np.load(path_y)
            return X, angles, y, ids
        else:
            return X, angles, ids

    # Process from scratch
    if mode == "train":
        json_path = os.path.join(Config.INPUT_DIR, Config.TRAIN_JSON)
        X, angles, y, ids = process_json_to_numpy(json_path, is_train=True)

        # Save to cache
        np.save(path_X, X)
        np.save(path_angles, angles)
        np.save(path_y, y)
        np.save(path_ids, ids)

        return X, angles, y, ids
    else:
        json_path = os.path.join(Config.INPUT_DIR, Config.TEST_JSON)
        X, angles, ids = process_json_to_numpy(json_path, is_train=False)

        # Save to cache
        np.save(path_X, X)
        np.save(path_angles, angles)
        np.save(path_ids, ids)

        return X, angles, ids


class IcebergDataset(Dataset):
    def __init__(self, X, angles, ids, y=None, transform=None, angle_imputer=None):
        """
        Args:
            X (np.ndarray): Images (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles (N,).
            ids (np.ndarray): Image IDs (N,).
            y (np.ndarray, optional): Labels (N,).
            transform (callable, optional): Augmentations.
            angle_imputer (float, optional): Value to replace NaNs in angles.
        """
        self.X = X
        self.angles = angles
        self.ids = ids
        self.y = y
        self.transform = transform
        self.angle_imputer = angle_imputer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get image
        img = self.X[idx]  # (3, 75, 75)

        # Apply transforms (Augmentation)
        # Note: torchvision transforms usually expect (C, H, W) tensor or PIL
        # Here we work with tensors.
        img_tensor = torch.from_numpy(img)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Handle Angle
        angle = self.angles[idx]
        if np.isnan(angle):
            if self.angle_imputer is not None:
                angle = self.angle_imputer
            else:
                angle = 0.0  # Fallback, though imputer should be provided

        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Return
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor, self.ids[idx]


def get_fold_loaders(fold_idx, load_cached_data=True):
    """
    Creates train and validation DataLoaders for a specific fold.
    Performs leak-free angle imputation.
    """
    # Load all training data
    X, angles, y, ids = load_dataset_with_cache(
        mode="train", load_cached_data=load_cached_data
    )

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Iterate to find the specific fold
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(skf.split(X, y)):
        if i == fold_idx:
            train_idx = t_idx
            val_idx = v_idx
            break

    if train_idx is None:
        raise ValueError(
            f"Fold index {fold_idx} out of range for {Config.N_FOLDS} folds."
        )

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angles_train, angles_val = angles[train_idx], angles[val_idx]
    ids_train, ids_val = ids[train_idx], ids[val_idx]

    # Calculate median angle on TRAINING set only (Leak-Free)
    angle_imputer = np.nanmedian(angles_train)

    # Define Augmentations
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train,
        angles_train,
        ids_train,
        y_train,
        transform=train_transform,
        angle_imputer=angle_imputer,
    )

    val_dataset = IcebergDataset(
        X_val, angles_val, ids_val, y_val, transform=None, angle_imputer=angle_imputer
    )

    # Create DataLoaders
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


def get_test_loader(load_cached_data=True):
    """
    Creates the test DataLoader.
    Uses the global training median for angle imputation.
    """
    # Load test data
    X_test, angles_test, ids_test = load_dataset_with_cache(
        mode="test", load_cached_data=load_cached_data
    )

    # Load train data just to compute global median for imputation
    _, angles_train, _, _ = load_dataset_with_cache(
        mode="train", load_cached_data=load_cached_data
    )
    global_angle_imputer = np.nanmedian(angles_train)

    test_dataset = IcebergDataset(
        X_test,
        angles_test,
        ids_test,
        y=None,
        transform=None,
        angle_imputer=global_angle_imputer,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader
