import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_FOLDS,
    IMAGE_SIZE,
)
from library.utils import get_logger

logger = get_logger(__name__)


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, X, angles, y=None, ids=None, transform=None, mode="train"):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): Image IDs of shape (N,).
            transform (callable, optional): Transform to be applied on a sample.
            mode (str): 'train' (returns label) or 'test' (returns id).
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.ids = ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to tensor. X is already (C, H, W) from processing
        img = torch.from_numpy(self.X[idx]).float()
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        if self.mode == "train" or self.mode == "val":
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle, label
        else:
            image_id = self.ids[idx]
            return img, angle, image_id


def process_data(load_cached_data=True):
    """
    Loads raw data, processes it (reshape, impute, stack), and caches it.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        dict: Contains 'X_train', 'y_train', 'angles_train', 'ids_train',
                       'X_test', 'angles_test', 'ids_test'
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache file paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "angles_train": os.path.join(WORKING_DIR, "angles_train.npy"),
        "ids_train": os.path.join(WORKING_DIR, "ids_train.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.npy"),
        "angles_test": os.path.join(WORKING_DIR, "angles_test.npy"),
        "ids_test": os.path.join(WORKING_DIR, "ids_test.npy"),
    }

    # Try loading from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            logger.info("Loading processed data from cache...")
            data = {}
            for key, path in cache_files.items():
                data[key] = np.load(path, allow_pickle=True)
            return data
        else:
            logger.info("Cache missing or incomplete. Processing from scratch...")
    else:
        logger.info("Forcing data processing from scratch...")

    # Load Raw Data
    train_path = os.path.join(INPUT_DIR, "train.json")
    test_path = os.path.join(INPUT_DIR, "test.json")

    logger.info("Reading JSON files...")
    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)

    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # Helper to process images
    def process_images(df):
        # Flattened lists to arrays
        b1 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
        )
        b2 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
        )

        # Average band
        b_avg = (b1 + b2) / 2.0

        # Stack to (N, 3, 75, 75) - Channel First
        X = np.stack([b1, b2, b_avg], axis=1)
        return X

    logger.info("Processing images...")
    X_train = process_images(df_train)
    X_test = process_images(df_test)

    # Process Angles (Imputation)
    logger.info("Processing incidence angles...")
    # Replace 'na' with NaN and convert to float
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")

    # Calculate median from training set (ignoring NaNs)
    angle_median = df_train["inc_angle"].median()

    # Impute
    df_train["inc_angle"] = df_train["inc_angle"].fillna(angle_median)
    df_test["inc_angle"] = df_test["inc_angle"].fillna(angle_median)

    angles_train = df_train["inc_angle"].values.astype(np.float32)
    angles_test = df_test["inc_angle"].values.astype(np.float32)

    # Labels and IDs
    y_train = df_train["is_iceberg"].values.astype(np.float32)
    ids_train = df_train["id"].values
    ids_test = df_test["id"].values

    # Save to Cache
    logger.info("Saving processed data to cache...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angles_train"], angles_train)
    np.save(cache_files["ids_train"], ids_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angles_test"], angles_test)
    np.save(cache_files["ids_test"], ids_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "angles_train": angles_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "angles_test": angles_test,
        "ids_test": ids_test,
    }


def get_loaders(fold=0, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold using Stratified K-Fold.

    Args:
        fold (int): The fold index (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    data = process_data(load_cached_data=load_cached_data)

    X = data["X_train"]
    y = data["y_train"]
    angles = data["angles_train"]
    ids = data["ids_train"]

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # Generate splits
    splits = list(skf.split(X, y))

    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range for {NUM_FOLDS} splits.")

    train_idx, val_idx = splits[fold]

    # Split data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    ang_train_fold, ang_val_fold = angles[train_idx], angles[val_idx]

    # Define Transforms (Only for Training)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train_fold,
        ang_train_fold,
        y_train_fold,
        transform=train_transform,
        mode="train",
    )
    val_dataset = IcebergDataset(
        X_val_fold, ang_val_fold, y_val_fold, transform=None, mode="val"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for the test set.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        DataLoader: test_loader
    """
    data = process_data(load_cached_data=load_cached_data)

    X_test = data["X_test"]
    angles_test = data["angles_test"]
    ids_test = data["ids_test"]

    test_dataset = IcebergDataset(
        X_test, angles_test, ids=ids_test, transform=None, mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
