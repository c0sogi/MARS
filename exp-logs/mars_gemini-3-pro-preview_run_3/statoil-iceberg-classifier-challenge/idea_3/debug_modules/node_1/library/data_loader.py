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
    Custom Dataset for Iceberg/Ship classification.
    Returns:
        img (Tensor): (3, 75, 75) image tensor.
        angle (Tensor): Scalar incidence angle.
        label (Tensor, optional): Target label (0 or 1).
    """

    def __init__(self, X, angles, y=None, transform=None):
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img_np = self.X[idx]
        angle_val = self.angles[idx]

        # Convert to Tensor
        # Input is (3, 75, 75), float32
        img = torch.from_numpy(img_np).float()
        angle = torch.tensor(angle_val, dtype=torch.float32)

        # Apply augmentations
        if self.transform:
            img = self.transform(img)

        # Return with label if available
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle, label
        else:
            return img, angle


def process_json_data(json_path, is_train=True):
    """
    Reads JSON, processes bands into images, handles missing angles.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Process Image Bands
    # Reshape flattened lists (5625) -> (75, 75)
    # Band 1: HH, Band 2: HV
    b1 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
    )
    b2 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
    )

    # Band 3: Average of HH and HV
    b3 = (b1 + b2) / 2.0

    # Stack channels: (N, 3, 75, 75)
    X = np.stack([b1, b2, b3], axis=1)

    # Process Incidence Angles
    # Convert 'na' to NaN, then impute with mean
    df["inc_angle"] = pd.to_numeric(df["inc_angle"], errors="coerce")
    angle_mean = df["inc_angle"].mean()
    df["inc_angle"] = df["inc_angle"].fillna(angle_mean)
    angles = df["inc_angle"].values.astype(np.float32)

    if is_train:
        y = df["is_iceberg"].values.astype(np.float32)
        return X, angles, y
    else:
        ids = df["id"].values
        return X, angles, ids


def load_data(mode="train", load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes raw JSON.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    if mode == "train":
        cache_X = Config.CACHE_X_TRAIN
        cache_angle = Config.CACHE_ANGLE_TRAIN
        cache_y = Config.CACHE_Y_TRAIN
        source_path = Config.TRAIN_JSON

        # Check Cache
        if (
            load_cached_data
            and os.path.exists(cache_X)
            and os.path.exists(cache_angle)
            and os.path.exists(cache_y)
        ):
            X = np.load(cache_X)
            angles = np.load(cache_angle)
            y = np.load(cache_y)
        else:
            # Process and Cache
            X, angles, y = process_json_data(source_path, is_train=True)
            np.save(cache_X, X)
            np.save(cache_angle, angles)
            np.save(cache_y, y)

        return X, angles, y

    elif mode == "test":
        cache_X = Config.CACHE_X_TEST
        cache_angle = Config.CACHE_ANGLE_TEST
        cache_ids = Config.CACHE_TEST_IDS
        source_path = Config.TEST_JSON

        # Check Cache
        if (
            load_cached_data
            and os.path.exists(cache_X)
            and os.path.exists(cache_angle)
            and os.path.exists(cache_ids)
        ):
            X = np.load(cache_X)
            angles = np.load(cache_angle)
            ids = np.load(cache_ids, allow_pickle=True)
        else:
            # Process and Cache
            X, angles, ids = process_json_data(source_path, is_train=False)
            np.save(cache_X, X)
            np.save(cache_angle, angles)
            np.save(cache_ids, ids)

        return X, angles, ids


def get_loaders(fold_idx=0, load_cached_data=True):
    """
    Creates Train and Validation DataLoaders for a specific K-Fold.
    """
    # Load all training data
    X, angles, y = load_data(mode="train", load_cached_data=load_cached_data)

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(X, y))

    if fold_idx >= len(splits):
        raise ValueError(
            f"Fold index {fold_idx} out of range for {Config.NUM_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_idx]

    # Subset data
    X_train, X_val = X[train_idx], X[val_idx]
    angle_train, angle_val = angles[train_idx], angles[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Debug Mode: Reduce dataset size
    if Config.DEBUG:
        limit = Config.DEBUG_SAMPLES
        X_train, angle_train, y_train = (
            X_train[:limit],
            angle_train[:limit],
            y_train[:limit],
        )
        X_val, angle_val, y_val = X_val[:limit], angle_val[:limit], y_val[:limit]

    # Define Augmentations (Train only)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angle_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angle_val, y_val, transform=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates Test DataLoader and returns test IDs.
    """
    X, angles, ids = load_data(mode="test", load_cached_data=load_cached_data)

    test_dataset = IcebergDataset(X, angles, y=None, transform=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, ids
