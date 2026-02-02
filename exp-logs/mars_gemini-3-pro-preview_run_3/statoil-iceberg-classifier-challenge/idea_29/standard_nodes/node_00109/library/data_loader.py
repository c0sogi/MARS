import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library import config, utils


def process_json_data(json_path, is_train=True):
    """
    Reads JSON file, constructs 3-channel images, extracts angles and labels/ids.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Process Images
    # Band 1 (HH) and Band 2 (HV)
    b1 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
    )
    b2 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
    )

    # Band 3 (Average)
    b3 = (b1 + b2) / 2.0

    # Stack to (N, 3, 75, 75)
    # Use np.stack on axis 1 for channel dimension
    X = np.stack([b1, b2, b3], axis=1)

    # Process Angles
    # Coerce 'na' to NaN
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    if is_train:
        y = df["is_iceberg"].values.astype(np.float32)
        ids = df["id"].values
        return X, angles, y, ids
    else:
        ids = df["id"].values
        return X, angles, ids


def load_and_process_data(load_cached_data=True):
    """
    Loads data from cache or raw JSON files.
    Implements imputation logic and caching.
    """
    utils.seed_everything(config.SEED)

    cache_dir = config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # File paths for cache
    cache_files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "angles_train": os.path.join(cache_dir, "angles_train.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "angles_test": os.path.join(cache_dir, "angles_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        angles_train = np.load(cache_files["angles_train"])
        X_test = np.load(cache_files["X_test"])
        angles_test = np.load(cache_files["angles_test"])
        ids_test = np.load(cache_files["ids_test"], allow_pickle=True)
        return X_train, angles_train, y_train, X_test, angles_test, ids_test

    print("Processing raw data from JSON...")

    # Load and process Train
    X_train, angles_train, y_train, _ = process_json_data(
        config.TRAIN_JSON_PATH, is_train=True
    )

    # Load and process Test
    X_test, angles_test, ids_test = process_json_data(
        config.TEST_JSON_PATH, is_train=False
    )

    # Impute Missing Angles
    # Compute median from valid training angles
    valid_angle_mask = ~np.isnan(angles_train)
    angle_median = np.median(angles_train[valid_angle_mask])

    # Fill NaNs in Train
    angles_train[np.isnan(angles_train)] = angle_median

    # Fill NaNs in Test
    angles_test[np.isnan(angles_test)] = angle_median

    # Save to cache
    print("Saving processed data to cache...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["angles_train"], angles_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angles_test"], angles_test)
    np.save(cache_files["ids_test"], ids_test)

    return X_train, angles_train, y_train, X_test, angles_test, ids_test


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=None):
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get image (C, H, W)
        img = self.X[idx]
        angle = self.angles[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


def get_loaders(fold_index, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold in 5-Fold CV.
    Also returns a Test DataLoader.
    """
    utils.seed_everything(config.SEED)

    # Load Data
    X_train_full, angles_train_full, y_train_full, X_test, angles_test, ids_test = (
        load_and_process_data(load_cached_data)
    )

    # Filter to include ONLY the training subset defined in metadata to prevent leakage.
    # Cite debug_lesson_7
    if os.path.exists(config.TRAIN_META_PATH):
        train_meta = pd.read_csv(config.TRAIN_META_PATH)
        train_indices = train_meta["original_index"].values
        X_train_full = X_train_full[train_indices]
        angles_train_full = angles_train_full[train_indices]
        y_train_full = y_train_full[train_indices]

    # Debug mode
    if config.DEBUG:
        subset_size = min(config.DEBUG_SUBSET_SIZE, len(X_train_full))
        X_train_full = X_train_full[:subset_size]
        angles_train_full = angles_train_full[:subset_size]
        y_train_full = y_train_full[:subset_size]

        X_test = X_test[:subset_size]
        angles_test = angles_test[:subset_size]
        ids_test = ids_test[:subset_size]

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Get indices for the requested fold
    splits = list(skf.split(X_train_full, y_train_full))
    train_idx, val_idx = splits[fold_index]

    # Create subsets
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    ang_train, ang_val = angles_train_full[train_idx], angles_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    # Define Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, ang_val, y_val, transform=None)
    test_dataset = IcebergDataset(X_test, angles_test, y=None, transform=None)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, ids_test
