import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import set_seed


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=None):
        """
        Custom Dataset for Iceberg vs Ship classification.

        Args:
            images (np.ndarray): Image data of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            labels (np.ndarray, optional): Target labels of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert numpy array to torch tensor
        # Input image is (3, 75, 75), float32
        image = torch.from_numpy(self.images[idx]).float()
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Apply augmentations if provided
        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "angle": angle}

        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return sample


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays (creating 3-channel images),
    and caches the results to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: Dictionary containing processed numpy arrays for train and test sets.
    """
    # Define cache paths
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "angle_train": os.path.join(Config.CACHE_DIR, "angle_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "ids_train": os.path.join(Config.CACHE_DIR, "ids_train.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "angle_test": os.path.join(Config.CACHE_DIR, "angle_test.npy"),
        "ids_test": os.path.join(Config.CACHE_DIR, "ids_test.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        data = {}
        for k, v in cache_files.items():
            data[k] = np.load(v)
        return data

    print("Processing raw data from scratch...")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Helper to process a list of records
    def process_records(records, is_train=True):
        ids = []
        band_1 = []
        band_2 = []
        angles = []
        labels = []

        for item in records:
            ids.append(item["id"])
            band_1.append(item["band_1"])
            band_2.append(item["band_2"])
            angles.append(item["inc_angle"])
            if is_train:
                labels.append(item["is_iceberg"])

        # Reshape bands to (N, 75, 75)
        b1 = np.array(band_1, dtype=np.float32).reshape(-1, 75, 75)
        b2 = np.array(band_2, dtype=np.float32).reshape(-1, 75, 75)

        # Create Band 3: Average of HH and HV
        b3 = (b1 + b2) / 2.0

        # Stack to create (N, 3, 75, 75)
        X = np.stack([b1, b2, b3], axis=1)

        # Process angles: Convert 'na' to NaN
        clean_angles = []
        for a in angles:
            if isinstance(a, str) and a == "na":
                clean_angles.append(np.nan)
            else:
                clean_angles.append(float(a))
        angle_arr = np.array(clean_angles, dtype=np.float32)

        ids_arr = np.array(ids)

        if is_train:
            y_arr = np.array(labels, dtype=np.float32)
            return X, angle_arr, ids_arr, y_arr
        else:
            return X, angle_arr, ids_arr, None

    # Load and process Train
    with open(Config.TRAIN_JSON, "r") as f:
        train_records = json.load(f)
    X_train, angle_train, ids_train, y_train = process_records(
        train_records, is_train=True
    )

    # Load and process Test
    with open(Config.TEST_JSON, "r") as f:
        test_records = json.load(f)
    X_test, angle_test, ids_test, _ = process_records(test_records, is_train=False)

    # Save to cache
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["angle_train"], angle_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["ids_train"], ids_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["angle_test"], angle_test)
    np.save(cache_files["ids_test"], ids_test)

    return {
        "X_train": X_train,
        "angle_train": angle_train,
        "y_train": y_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "angle_test": angle_test,
        "ids_test": ids_test,
    }


def get_loaders(fold_idx, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold using Stratified K-Fold.
    Performs leak-free imputation of incidence angles.

    Args:
        fold_idx (int): Index of the fold (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load data
    data = process_and_cache_data(load_cached_data=load_cached_data)
    X = data["X_train"]
    angles = data["angle_train"]
    y = data["y_train"]

    # Debug subsampling
    if Config.DEBUG:
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(X))
        indices = np.random.choice(len(X), subset_size, replace=False)
        X = X[indices]
        angles = angles[indices]
        y = y[indices]

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(X, y))

    if not (0 <= fold_idx < Config.NUM_FOLDS):
        raise ValueError(f"Fold index {fold_idx} out of range (0-{Config.NUM_FOLDS-1})")

    train_idx, val_idx = splits[fold_idx]

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angle_train_raw, angle_val_raw = angles[train_idx], angles[val_idx]

    # Leak-Free Imputation: Compute median on TRAIN only
    train_median = np.nanmedian(angle_train_raw)

    # Fill NaNs
    angle_train_filled = np.nan_to_num(angle_train_raw, nan=train_median)
    angle_val_filled = np.nan_to_num(angle_val_raw, nan=train_median)

    # Augmentations (Train only)
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angle_train_filled, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angle_val_filled, y_val, transform=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
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
    Creates a DataLoader for the test set.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (test_loader, test_ids)
    """
    data = process_and_cache_data(load_cached_data=load_cached_data)
    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # Handle missing angles in test set (if any)
    # Using global median of the test set itself if needed, or 0.
    # The problem statement implies NaNs are mostly in train, but we handle it safely.
    if np.isnan(angle_test).any():
        median = np.nanmedian(angle_test)
        angle_test = np.nan_to_num(angle_test, nan=median)

    test_dataset = IcebergDataset(X_test, angle_test, labels=None, transform=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader, ids_test
