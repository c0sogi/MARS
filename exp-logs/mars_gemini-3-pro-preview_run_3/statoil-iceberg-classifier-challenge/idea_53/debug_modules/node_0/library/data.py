import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import get_logger

# Initialize Logger
logger = get_logger("data_module")


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles 3-channel image construction and incidence angle integration.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75), float32.
            angles (np.ndarray): Shape (N,), float32.
            labels (np.ndarray, optional): Shape (N,), int/float.
            ids (np.ndarray, optional): Shape (N,), string.
            transform (callable, optional): Augmentation pipeline.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert to tensor
        # images are already (C, H, W) from preprocessing
        image = torch.from_numpy(self.images[idx]).float()
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Apply augmentations (only works on tensors or PIL images)
        if self.transform:
            image = self.transform(image)

        # Return tuple based on availability of labels/ids
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            # Inference mode
            img_id = self.ids[idx] if self.ids is not None else ""
            return image, angle, img_id


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays, and caches the result.
    Implements median imputation for incidence angles and standard scaling for images.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: Contains 'X_train', 'y_train', 'angle_train', 'ids_train',
                       'X_test', 'angle_test', 'ids_test'
    """
    # Define cache paths
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "angle_train": os.path.join(Config.CACHE_DIR, "angle_train.npy"),
        "ids_train": os.path.join(Config.CACHE_DIR, "ids_train.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "angle_test": os.path.join(Config.CACHE_DIR, "angle_test.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
    }

    # 1. Try loading from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            logger.info(f"Loading processed data from cache: {Config.CACHE_DIR}")
            data = {k: np.load(v, allow_pickle=True) for k, v in cache_files.items()}
            return data
        else:
            logger.info("Cache miss or partial cache. Processing from scratch...")
    else:
        logger.info("Force processing from scratch...")

    # 2. Process from scratch
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Load Train Data
    logger.info(f"Loading raw train data from {Config.TRAIN_JSON}...")
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    df_train = pd.DataFrame(train_data)

    # Load Test Data
    logger.info(f"Loading raw test data from {Config.TEST_JSON}...")
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)
    df_test = pd.DataFrame(test_data)

    # Debugging subset
    if Config.DEBUG:
        logger.info(f"DEBUG mode: subsampling {Config.DEBUG_SUBSET_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    # --- Image Processing ---
    def process_images(df):
        # Flattened list to (N, 75, 75)
        b1 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
        )
        b2 = np.array(
            [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
        )

        # Synthetic 3rd band: Average of HH and HV
        b3 = (b1 + b2) / 2.0

        # Stack to (N, 3, 75, 75) - Channel First for PyTorch
        # axis=1 creates the channel dimension
        images = np.stack([b1, b2, b3], axis=1)
        return images

    logger.info("Processing training images...")
    X_train = process_images(df_train)
    logger.info("Processing test images...")
    X_test = process_images(df_test)

    # --- Normalization ---
    # Compute stats on training set
    # Mean and Std per channel: (3, 1, 1) to broadcast
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True)

    logger.info(f"Normalization Stats - Mean: {mean.flatten()}, Std: {std.flatten()}")

    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # --- Angle Processing ---
    def process_angles(df, is_train=True, median_val=None):
        # Convert to numeric, coerce errors to NaN
        angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(
            np.float32
        )

        if is_train:
            # Calculate median from valid values
            valid_mask = ~np.isnan(angles)
            median_val = np.median(angles[valid_mask])
            logger.info(f"Imputing missing angles with median: {median_val:.4f}")

        # Impute
        angles[np.isnan(angles)] = median_val
        return angles, median_val

    angle_train, median_angle = process_angles(df_train, is_train=True)
    angle_test, _ = process_angles(df_test, is_train=False, median_val=median_angle)

    # --- Labels and IDs ---
    y_train = df_train["is_iceberg"].values.astype(np.float32)
    ids_train = df_train["id"].values
    ids_test = df_test["id"].values

    # 3. Save to cache
    logger.info("Saving processed data to cache...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["ids_train"], ids_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angle_test"], angle_test)
    np.save(cache_files["ids_test"], ids_test)

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "angle_train": angle_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "angle_test": angle_test,
        "ids_test": ids_test,
    }
    return data


def get_folds(y, n_folds=5, seed=42):
    """
    Generates Stratified K-Fold indices.

    Args:
        y (np.ndarray): Target labels.
        n_folds (int): Number of folds.
        seed (int): Random seed.

    Returns:
        generator: Yields (train_index, val_index).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return skf.split(np.zeros(len(y)), y)


def get_dataloaders(data, fold_idx, batch_size, num_workers=0):
    """
    Creates DataLoaders for a specific fold.

    Args:
        data (dict): Dictionary containing processed data arrays.
        fold_idx (int): The current fold index (0 to NUM_FOLDS-1).
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.

    Returns:
        tuple: (train_loader, val_loader)
    """
    X = data["X_train"]
    y = data["y_train"]
    angles = data["angle_train"]
    ids = data["ids_train"]

    # Get split indices
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(X, y))

    if fold_idx >= len(splits):
        raise ValueError(
            f"Fold index {fold_idx} out of range for {Config.NUM_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_idx]

    # Create subsets
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    ang_train, ang_val = angles[train_idx], angles[val_idx]
    ids_train_sub, ids_val_sub = ids[train_idx], ids[val_idx]

    # Augmentations
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, ids_train_sub, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, ang_val, y_val, ids_val_sub, transform=None)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(data, batch_size, num_workers=0):
    """
    Creates DataLoader for the test set.
    """
    test_dataset = IcebergDataset(
        data["X_test"],
        data["angle_test"],
        labels=None,
        ids=data["ids_test"],
        transform=None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    return test_loader
