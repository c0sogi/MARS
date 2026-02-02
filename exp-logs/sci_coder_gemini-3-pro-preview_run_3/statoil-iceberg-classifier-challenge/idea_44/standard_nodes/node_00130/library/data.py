import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library import config, utils

# Define constants
CACHE_DIR = config.CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images: (N, 3, 75, 75) numpy array
            angles: (N,) numpy array
            labels: (N,) numpy array (optional)
            transform: torchvision transforms
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert to tensor (C, H, W)
        # Images are already float32, (3, 75, 75)
        image = torch.from_numpy(self.images[idx]).float()
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            return image, angle


def _process_json_data(json_path, metadata_df, is_test=False):
    """
    Reads json data and aligns it with metadata_df.
    Returns X (images), angles, y (labels, if not test), ids
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create a dictionary for fast lookup by id
    data_dict = {item["id"]: item for item in data}

    # Initialize arrays
    num_samples = len(metadata_df)
    X = np.zeros((num_samples, 3, 75, 75), dtype=np.float32)
    angles = np.zeros(num_samples, dtype=np.float32)
    ids = []

    if not is_test:
        y = np.zeros(num_samples, dtype=np.float32)
    else:
        y = None

    for i, row in metadata_df.iterrows():
        img_id = row["id"]
        item = data_dict[img_id]

        # Process Bands
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        avg = (b1 + b2) / 2.0

        # Stack channels (C, H, W)
        X[i, 0, :, :] = b1
        X[i, 1, :, :] = b2
        X[i, 2, :, :] = avg

        # Process Angle
        # Metadata already handles 'na' conversion to NaN
        angles[i] = row["inc_angle"]

        ids.append(img_id)

        if not is_test:
            y[i] = row["is_iceberg"]

    return X, angles, y, np.array(ids)


def load_dataset_arrays(load_cached_data=True):
    """
    Loads dataset arrays, using cache if available and requested.
    Combines train and val metadata for the full training set.
    """
    # File paths for cache
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "angles_train": os.path.join(CACHE_DIR, "angles_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "ids_train": os.path.join(CACHE_DIR, "ids_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "angles_test": os.path.join(CACHE_DIR, "angles_test.npy"),
        "ids_test": os.path.join(CACHE_DIR, "ids_test.npy"),
    }

    # Check if cache exists
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

        return (X_train, angles_train, y_train, ids_train), (
            X_test,
            angles_test,
            ids_test,
        )

    print("Processing data from scratch...")

    # Load Metadata
    df_train = pd.read_csv(config.TRAIN_META_PATH)
    df_val = pd.read_csv(config.VAL_META_PATH)
    # Concatenate to form full training set for CV
    df_full_train = pd.concat([df_train, df_val], ignore_index=True)
    df_test = pd.read_csv(config.TEST_META_PATH)

    # Process Train
    X_train, angles_train, y_train, ids_train = _process_json_data(
        config.TRAIN_JSON, df_full_train, is_test=False
    )

    # Process Test
    X_test, angles_test, _, ids_test = _process_json_data(
        config.TEST_JSON, df_test, is_test=True
    )

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["angles_train"], angles_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["ids_train"], ids_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angles_test"], angles_test)
    np.save(cache_files["ids_test"], ids_test)

    return (X_train, angles_train, y_train, ids_train), (X_test, angles_test, ids_test)


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Returns train and validation dataloaders for a specific fold.
    Performs Stratified K-Fold splitting and incidence angle imputation.
    """
    utils.set_seed(config.SEED)

    # Load all data
    (X, angles, y, ids), _ = load_dataset_arrays(load_cached_data)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Get indices for the requested fold
    # skf.split returns generator, we iterate to find the specific fold
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if i == fold_idx:
            break
    else:
        raise ValueError(f"Fold index {fold_idx} out of range (0-{config.NUM_FOLDS-1})")

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angles_train, angles_val = angles[train_idx], angles[val_idx]

    # Impute missing angles with training median
    # Identify valid angles in training set
    valid_angles = angles_train[~np.isnan(angles_train)]
    if len(valid_angles) > 0:
        fill_value = np.median(valid_angles)
    else:
        fill_value = 0.0  # Fallback

    angles_train = np.where(np.isnan(angles_train), fill_value, angles_train)
    # Use training median for validation set to prevent leakage
    angles_val = np.where(np.isnan(angles_val), fill_value, angles_val)

    # Define Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angles_val, y_val, transform=None)

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Returns test dataloader.
    Imputes missing angles using the median from the full training set.
    """
    utils.set_seed(config.SEED)

    (X_train_all, angles_train_all, _, _), (X_test, angles_test, ids_test) = (
        load_dataset_arrays(load_cached_data)
    )

    # Calculate median from full training set for imputation
    valid_angles = angles_train_all[~np.isnan(angles_train_all)]
    if len(valid_angles) > 0:
        fill_value = np.median(valid_angles)
    else:
        fill_value = 0.0

    angles_test = np.where(np.isnan(angles_test), fill_value, angles_test)

    test_dataset = IcebergDataset(X_test, angles_test, labels=None, transform=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_loader, ids_test
