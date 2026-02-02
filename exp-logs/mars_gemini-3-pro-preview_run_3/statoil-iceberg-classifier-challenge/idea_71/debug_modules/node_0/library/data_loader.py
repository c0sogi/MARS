import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

from library.config import (
    SEED,
    BATCH_SIZE,
    NUM_FOLDS,
    CACHE_DIR,
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META,
    VAL_META,
    TEST_META,
)
from library.utils import set_seed

# Ensure deterministic behavior for transforms where possible
set_seed(SEED)


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles 3-channel input (HH, HV, Avg) and incidence angles.
    """

    def __init__(self, X, y, angles, transform=None):
        """
        Args:
            X (np.ndarray): Image data of shape (N, 3, 75, 75).
            y (np.ndarray, optional): Target labels of shape (N,). None for test set.
            angles (np.ndarray): Incidence angles of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.y = y
        self.angles = angles
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert numpy array to torch tensor
        img = torch.from_numpy(self.X[idx])  # Shape: (3, 75, 75)
        angle = self.angles[idx]

        # Apply augmentations if provided
        if self.transform:
            img = self.transform(img)

        # Convert angle to tensor
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle_tensor, label
        else:
            return img, angle_tensor


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays (3 channels), and caches it.

    Logic:
        1. Checks if .npy files exist in CACHE_DIR.
        2. If load_cached_data=True and files exist, loads and returns them.
        3. Otherwise, processes raw JSONs:
           - Reshapes flattened bands to 75x75.
           - Creates 3rd band (Avg of HH and HV).
           - Stacks to (3, 75, 75).
           - Parses incidence angles (converting 'na' to NaN).
        4. Saves processed arrays to CACHE_DIR.

    Returns:
        X_train (np.ndarray): Training images (N_train, 3, 75, 75).
        y_train (np.ndarray): Training labels (N_train,).
        angle_train (np.ndarray): Training angles (N_train,) with NaNs.
        X_test (np.ndarray): Test images (N_test, 3, 75, 75).
        ids_test (np.ndarray): Test IDs (N_test,).
        angle_test (np.ndarray): Test angles (N_test,) with NaNs.
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "angle_train": os.path.join(CACHE_DIR, "angle_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "ids_test": os.path.join(CACHE_DIR, "ids_test.npy"),
        "angle_test": os.path.join(CACHE_DIR, "angle_test.npy"),
    }

    # Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(f) for f in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            return (
                np.load(cache_files["X_train"]),
                np.load(cache_files["y_train"]),
                np.load(cache_files["angle_train"]),
                np.load(cache_files["X_test"]),
                np.load(cache_files["ids_test"]),
                np.load(cache_files["angle_test"]),
            )

    print("Processing data from scratch...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load Metadata
    # We combine train and val metadata to reconstruct the full training set for CV
    train_meta = pd.read_csv(TRAIN_META)
    val_meta = pd.read_csv(VAL_META)
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)
    test_meta = pd.read_csv(TEST_META)

    # Load Raw JSONs
    print(f"Loading {TRAIN_JSON}...")
    with open(TRAIN_JSON, "r") as f:
        raw_train = json.load(f)
    # Create lookup map for speed
    train_map = {item["id"]: item for item in raw_train}

    print(f"Loading {TEST_JSON}...")
    with open(TEST_JSON, "r") as f:
        raw_test = json.load(f)
    test_map = {item["id"]: item for item in raw_test}

    def _process_subset(df, data_map, is_train=True):
        X_list = []
        angles_list = []
        ids_list = []
        targets_list = []

        for _, row in df.iterrows():
            img_id = row["id"]
            item = data_map[img_id]

            # Process Bands
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            avg = (b1 + b2) / 2.0

            # Stack channels: (3, 75, 75)
            img = np.stack([b1, b2, avg], axis=0)
            X_list.append(img)

            # Process Angle
            ang = item["inc_angle"]
            if ang == "na":
                angles_list.append(np.nan)
            else:
                angles_list.append(float(ang))

            ids_list.append(img_id)

            if is_train:
                targets_list.append(row["is_iceberg"])

        X_arr = np.array(X_list, dtype=np.float32)
        ang_arr = np.array(angles_list, dtype=np.float32)
        ids_arr = np.array(ids_list)

        if is_train:
            y_arr = np.array(targets_list, dtype=np.float32)
            return X_arr, ang_arr, ids_arr, y_arr
        else:
            return X_arr, ang_arr, ids_arr, None

    # Process Train (Full)
    X_train, angle_train, _, y_train = _process_subset(
        full_train_meta, train_map, is_train=True
    )

    # Process Test
    X_test, angle_test, ids_test, _ = _process_subset(
        test_meta, test_map, is_train=False
    )

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["ids_test"], ids_test)
    np.save(cache_files["angle_test"], angle_test)

    print("Data processing complete and cached.")
    return X_train, y_train, angle_train, X_test, ids_test, angle_test


def get_fold_loaders(fold_idx, X_full, y_full, angle_full, X_test, angle_test):
    """
    Creates DataLoaders for a specific fold in Stratified K-Fold Cross Validation.
    Handles leak-free imputation of incidence angles.

    Args:
        fold_idx (int): The current fold index (0 to NUM_FOLDS-1).
        X_full (np.ndarray): All training images.
        y_full (np.ndarray): All training labels.
        angle_full (np.ndarray): All training angles (with NaNs).
        X_test (np.ndarray): Test images.
        angle_test (np.ndarray): Test angles (with NaNs).

    Returns:
        train_loader (DataLoader): Loader for training subset.
        val_loader (DataLoader): Loader for validation subset.
        test_loader (DataLoader): Loader for test set (with imputed angles).
    """
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Get indices for the specific fold
    # We iterate until we find the requested fold
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(skf.split(X_full, y_full)):
        if i == fold_idx:
            train_idx = t_idx
            val_idx = v_idx
            break

    if train_idx is None:
        raise ValueError(f"Fold index {fold_idx} out of range for {NUM_FOLDS} splits.")

    # Split Data
    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train, y_val = y_full[train_idx], y_full[val_idx]
    angle_train, angle_val = angle_full[train_idx], angle_full[val_idx]

    # Impute Missing Angles (Leak-Free)
    # Calculate median ONLY on training data for this fold
    train_angle_median = np.nanmedian(angle_train)

    # Fill NaNs in all sets using the training median
    angle_train_filled = np.nan_to_num(angle_train, nan=train_angle_median)
    angle_val_filled = np.nan_to_num(angle_val, nan=train_angle_median)
    angle_test_filled = np.nan_to_num(angle_test, nan=train_angle_median)

    # Define Transforms (Augmentation only for training)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Create Datasets
    train_ds = IcebergDataset(
        X_train, y_train, angle_train_filled, transform=train_transform
    )
    val_ds = IcebergDataset(X_val, y_val, angle_val_filled, transform=None)
    test_ds = IcebergDataset(X_test, None, angle_test_filled, transform=None)

    # Create DataLoaders
    # num_workers=2 is generally safe for these environments; pin_memory=True for GPU
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to maintain statistics stability
    )

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
