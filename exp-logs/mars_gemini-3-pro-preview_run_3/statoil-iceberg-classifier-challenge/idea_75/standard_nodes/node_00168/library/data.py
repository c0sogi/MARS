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
    PyTorch Dataset for Iceberg vs Ship classification.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75), float32.
            angles (np.ndarray): Shape (N,), float32.
            labels (np.ndarray, optional): Shape (N,), float32 (0 or 1).
            ids (np.ndarray, optional): Shape (N,), string IDs.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        image = torch.from_numpy(self.images[idx])  # (3, 75, 75)
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        # Apply transforms (Augmentation)
        if self.transform:
            image = self.transform(image)

        # Prepare return tuple
        # Structure: (inputs, target, id)
        # inputs is a tuple (image, angle)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            # Dummy label for test set
            label = torch.tensor(0.0, dtype=torch.float32)

        id_str = self.ids[idx] if self.ids is not None else ""

        return (image, angle), label, id_str


def _process_and_cache(json_path, cache_prefix, is_train=True):
    """
    Parses the raw JSON, processes bands, and saves to .npy cache.
    """
    print(f"Processing raw data from {json_path}...")

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Process Images
    # Band 1 and Band 2 are lists of floats. Reshape to (75, 75)
    # Stack to (N, 3, 75, 75) where ch3 = avg(ch1, ch2)

    # Convert lists to numpy arrays first for speed
    b1 = np.array([np.array(b).reshape(75, 75) for b in df["band_1"]])
    b2 = np.array([np.array(b).reshape(75, 75) for b in df["band_2"]])
    b3 = (b1 + b2) / 2.0

    # Stack along channel dimension (N, C, H, W)
    # b1, b2, b3 are (N, 75, 75)
    X = np.stack([b1, b2, b3], axis=1).astype(np.float32)

    # Process Angles
    # Replace 'na' with NaN and convert to float
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    # Process IDs
    ids = df["id"].values

    # Process Labels (only for train)
    if is_train:
        y = df["is_iceberg"].values.astype(np.float32)
    else:
        y = None

    # Save to cache
    np.save(os.path.join(Config.CACHE_DIR, f"X_{cache_prefix}.npy"), X)
    np.save(os.path.join(Config.CACHE_DIR, f"angle_{cache_prefix}.npy"), angles)
    np.save(os.path.join(Config.CACHE_DIR, f"ids_{cache_prefix}.npy"), ids)
    if is_train:
        np.save(os.path.join(Config.CACHE_DIR, f"y_{cache_prefix}.npy"), y)

    print(f"Cached data saved to {Config.CACHE_DIR} with prefix '{cache_prefix}'")
    return X, angles, y, ids


def _load_data(mode, load_cached_data=True):
    """
    Generic loader that handles caching logic.
    mode: 'train' or 'test'
    """
    Config.create_directories()

    prefix = mode
    json_path = Config.TRAIN_JSON if mode == "train" else Config.TEST_JSON

    # Define cache paths
    x_path = os.path.join(Config.CACHE_DIR, f"X_{prefix}.npy")
    ang_path = os.path.join(Config.CACHE_DIR, f"angle_{prefix}.npy")
    ids_path = os.path.join(Config.CACHE_DIR, f"ids_{prefix}.npy")
    y_path = os.path.join(Config.CACHE_DIR, f"y_{prefix}.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(x_path) and os.path.exists(ang_path) and os.path.exists(ids_path)
    )
    if mode == "train":
        cache_exists = cache_exists and os.path.exists(y_path)

    if load_cached_data and cache_exists:
        print(f"Loading {mode} data from cache...")
        X = np.load(x_path)
        angles = np.load(ang_path)
        ids = np.load(ids_path, allow_pickle=True)
        y = np.load(y_path) if mode == "train" else None
    else:
        # Process from scratch
        X, angles, y, ids = _process_and_cache(
            json_path, prefix, is_train=(mode == "train")
        )

    # Debug subset
    if Config.DEBUG:
        limit = min(Config.DEBUG_SUBSET_SIZE, len(X))
        print(f"DEBUG MODE: Slicing {mode} data to {limit} samples.")
        X = X[:limit]
        angles = angles[:limit]
        ids = ids[:limit]
        if y is not None:
            y = y[:limit]

    return X, angles, y, ids


def get_fold_loaders(fold_index, load_cached_data=True):
    """
    Returns train and validation DataLoaders for a specific fold.
    Implements leak-free angle imputation.
    """
    set_seed(Config.SEED)

    # Load all training data
    X, angles, y, ids = _load_data("train", load_cached_data)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    # We iterate to find the specific fold indices
    splits = list(skf.split(X, y))
    if fold_index >= len(splits):
        raise ValueError(
            f"Fold index {fold_index} out of range for {Config.NUM_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_index]

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angle_train, angle_val = angles[train_idx], angles[val_idx]
    ids_train, ids_val = ids[train_idx], ids[val_idx]

    # Leak-free Imputation
    # Calculate median ONLY on training data where angle is not NaN
    valid_angles = angle_train[~np.isnan(angle_train)]
    if len(valid_angles) > 0:
        fill_value = np.median(valid_angles)
    else:
        fill_value = 0.0  # Fallback, unlikely

    # Apply imputation
    angle_train = np.where(np.isnan(angle_train), fill_value, angle_train)
    angle_val = np.where(np.isnan(angle_val), fill_value, angle_val)

    # Define Transforms (Augmentation)
    # Only for training set
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angle_train, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angle_val, y_val, ids_val, transform=None)

    # Create Loaders
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

    print(
        f"Fold {fold_index}: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}"
    )
    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns the test DataLoader.
    Imputes test angles using the median of the full training set.
    """
    set_seed(Config.SEED)

    # Load Test Data
    X_test, angle_test, _, ids_test = _load_data("test", load_cached_data)

    # Load Train Data just for computing the median angle
    # We ignore the images/labels to save memory if possible, but _load_data loads all.
    # Given the dataset size (small), loading all is fine.
    _, angle_train_all, _, _ = _load_data("train", load_cached_data)

    # Compute global training median
    valid_train_angles = angle_train_all[~np.isnan(angle_train_all)]
    if len(valid_train_angles) > 0:
        fill_value = np.median(valid_train_angles)
    else:
        fill_value = 0.0

    # Impute Test Data
    angle_test = np.where(np.isnan(angle_test), fill_value, angle_test)

    # Create Dataset (No augmentation)
    test_dataset = IcebergDataset(
        X_test, angle_test, labels=None, ids=ids_test, transform=None
    )

    # Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Important: Must not shuffle test data for submission
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Test Set: {len(test_dataset)} samples.")
    return test_loader
