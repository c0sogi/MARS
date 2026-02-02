import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library import config, utils


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            ids (np.ndarray, optional): Shape (N,)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and convert to float tensor
        image = self.images[idx]
        image = torch.from_numpy(image).float()

        # Load angle
        angle = self.angles[idx]
        angle = torch.tensor(angle).float()

        # Apply augmentations if provided
        if self.transform:
            image = self.transform(image)

        # Return data based on mode (Train/Val vs Test)
        if self.labels is not None:
            label = self.labels[idx]
            # Return label as float for BCEWithLogitsLoss
            return image, angle, torch.tensor(label).float()
        else:
            # Test mode returns ID for submission generation
            id_val = self.ids[idx]
            return image, angle, id_val


def process_json_data(json_path, is_test=False):
    """
    Parses the JSON file and processes the bands into a 3-channel image.
    Channel 1: HH (band_1)
    Channel 2: HV (band_2)
    Channel 3: Average of HH and HV
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Pre-allocate arrays
    num_samples = len(data)
    images = np.zeros((num_samples, 3, 75, 75), dtype=np.float32)
    angles = np.zeros(num_samples, dtype=np.float32)
    ids = []
    labels = np.zeros(num_samples, dtype=np.float32) if not is_test else None

    for i, item in enumerate(data):
        # Process Images
        # Band 1 and Band 2 are flattened 5625 elements
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)

        # Channel 3: Average
        avg = (b1 + b2) / 2.0

        # Stack channels: (3, 75, 75)
        images[i, 0, :, :] = b1
        images[i, 1, :, :] = b2
        images[i, 2, :, :] = avg

        # Process Angle
        # "na" values are converted to NaN for later imputation
        angle_val = item["inc_angle"]
        if angle_val == "na":
            angles[i] = np.nan
        else:
            angles[i] = float(angle_val)

        # Process ID
        ids.append(item["id"])

        # Process Label (only for train)
        if not is_test:
            labels[i] = item["is_iceberg"]

    ids = np.array(ids)

    return images, angles, labels, ids


def load_processed_data(is_test=False, load_cached_data=True):
    """
    Handles caching logic. Loads from .npy if available and requested,
    otherwise processes from raw JSON and saves to cache.
    """
    # Define cache paths based on mode
    if is_test:
        path_X = config.CACHE_PATH_X_TEST
        path_angle = config.CACHE_PATH_ANGLE_TEST
        path_ids = config.CACHE_PATH_IDS_TEST
        # Test set has no labels
        path_y = None
        json_path = config.TEST_JSON
    else:
        # We use the TRAIN cache paths for the FULL labeled dataset
        path_X = config.CACHE_PATH_X_TRAIN
        path_y = config.CACHE_PATH_Y_TRAIN
        path_angle = config.CACHE_PATH_ANGLE_TRAIN
        path_ids = os.path.join(
            config.WORKING_DIR, "ids_train.npy"
        )  # Extra cache for IDs
        json_path = config.TRAIN_JSON

    # Check if cache exists
    cache_exists = (
        os.path.exists(path_X)
        and os.path.exists(path_angle)
        and (
            is_test
            and os.path.exists(path_ids)
            or (not is_test and os.path.exists(path_y))
        )
    )

    if load_cached_data and cache_exists:
        # Load from cache
        X = np.load(path_X)
        angles = np.load(path_angle)
        if is_test:
            ids = np.load(path_ids)
            y = None
        else:
            y = np.load(path_y)
            ids = np.load(path_ids) if os.path.exists(path_ids) else None
    else:
        # Process from scratch
        X, angles, y, ids = process_json_data(json_path, is_test=is_test)

        # Save to cache
        np.save(path_X, X)
        np.save(path_angle, angles)
        if is_test:
            np.save(path_ids, ids)
        else:
            np.save(path_y, y)
            np.save(path_ids, ids)

    return X, angles, y, ids


def get_data_loaders(fold_idx, load_cached_data=True):
    """
    Returns train and validation DataLoaders for a specific fold using Stratified K-Fold.
    Imputes missing incidence angles using the median of the training fold.
    """
    # Load full labeled dataset
    X, angles, y, _ = load_processed_data(
        is_test=False, load_cached_data=load_cached_data
    )

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Get indices for the requested fold
    # We iterate to find the specific fold
    fold_generator = skf.split(X, y)
    train_idx, val_idx = next(x for i, x in enumerate(fold_generator) if i == fold_idx)

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    angles_train, angles_val = angles[train_idx], angles[val_idx]

    # Impute missing incidence angles
    # Calculate median on TRAIN set only to avoid leakage
    angle_median = np.nanmedian(angles_train)

    # Fill NaNs
    angles_train[np.isnan(angles_train)] = angle_median
    angles_val[np.isnan(angles_val)] = angle_median

    # Define Transforms (Augmentation for Train only)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angles_val, y_val, transform=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns the test DataLoader.
    Imputes missing incidence angles using the global median of the training set.
    """
    # Load Test Data
    X_test, angles_test, _, ids_test = load_processed_data(
        is_test=True, load_cached_data=load_cached_data
    )

    # Load Train Data (only for calculating global median for imputation)
    _, angles_train, _, _ = load_processed_data(
        is_test=False, load_cached_data=load_cached_data
    )

    # Calculate global training median
    angle_median = np.nanmedian(angles_train)

    # Impute missing values in test set
    angles_test[np.isnan(angles_test)] = angle_median

    # Create Dataset
    test_dataset = IcebergDataset(
        X_test, angles_test, labels=None, ids=ids_test, transform=None
    )

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
