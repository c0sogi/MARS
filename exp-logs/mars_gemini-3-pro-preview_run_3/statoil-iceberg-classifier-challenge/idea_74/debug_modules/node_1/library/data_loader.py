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
        # Convert to torch tensor
        # Images are (3, 75, 75), float32
        image = torch.from_numpy(self.images[idx]).float()

        angle = self.angles[idx]
        if angle is None or np.isnan(angle):
            # This should have been handled by imputation before creating dataset,
            # but as a fallback we use 0.0 (though the loader logic ensures this doesn't happen)
            angle = 0.0
        angle = torch.tensor(angle, dtype=torch.float32)

        sample = {"image": image, "angle": angle}

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            sample["label"] = label

        # Apply transforms (Augmentation)
        # Note: transforms expect image as input.
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample


def _process_json_data(json_path, is_train=True):
    """
    Reads raw json, processes bands into (N, 3, 75, 75) images, extracts meta.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Process Images
    # Band 1 and Band 2 are lists of 5625 floats
    # We need to reshape to (75, 75)

    # Stack all band_1 into a matrix (N, 5625)
    b1_flat = np.stack(df["band_1"].values)
    b2_flat = np.stack(df["band_2"].values)

    # Reshape to (N, 75, 75)
    b1 = b1_flat.reshape(-1, 75, 75)
    b2 = b2_flat.reshape(-1, 75, 75)

    # Calculate Band 3 (Average)
    b3 = (b1 + b2) / 2.0

    # Stack into (N, 3, 75, 75)
    # Channel 0: HH, Channel 1: HV, Channel 2: Avg
    images = np.stack([b1, b2, b3], axis=1).astype(np.float32)

    # Process Angles
    # Convert 'na' to NaN and then to float
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").values.astype(np.float32)

    # IDs
    ids = df["id"].values

    labels = None
    if is_train:
        labels = df["is_iceberg"].values.astype(np.float32)

    return images, angles, labels, ids


def load_data(mode="train", load_cached_data=True):
    """
    Loads data, using cache if available and requested.

    Args:
        mode (str): 'train' or 'test'
        load_cached_data (bool): If True, try to load from .npy files.

    Returns:
        tuple: (images, angles, labels, ids) for train
               (images, angles, ids) for test
    """
    Config.setup_directories()

    cache_prefix = "train" if mode == "train" else "test"

    # Define cache paths
    path_X = os.path.join(Config.CACHE_DIR, f"X_{cache_prefix}.npy")
    path_ang = os.path.join(Config.CACHE_DIR, f"angle_{cache_prefix}.npy")
    path_ids = os.path.join(Config.CACHE_DIR, f"ids_{cache_prefix}.npy")
    path_y = os.path.join(Config.CACHE_DIR, f"y_{cache_prefix}.npy")  # Only for train

    # Check if cache exists
    cache_exists = (
        os.path.exists(path_X) and os.path.exists(path_ang) and os.path.exists(path_ids)
    )
    if mode == "train":
        cache_exists = cache_exists and os.path.exists(path_y)

    if load_cached_data and cache_exists:
        # Load from cache
        # print(f"Loading {mode} data from cache...")
        X = np.load(path_X)
        angles = np.load(path_ang)
        ids = np.load(path_ids)
        if mode == "train":
            y = np.load(path_y)
            return X, angles, y, ids
        else:
            return X, angles, ids

    # Process from scratch
    # print(f"Processing {mode} data from raw JSON...")
    json_path = Config.TRAIN_JSON if mode == "train" else Config.TEST_JSON

    if mode == "train":
        X, angles, y, ids = _process_json_data(json_path, is_train=True)
        np.save(path_X, X)
        np.save(path_ang, angles)
        np.save(path_ids, ids)
        np.save(path_y, y)
        return X, angles, y, ids
    else:
        X, angles, _, ids = _process_json_data(json_path, is_train=False)
        np.save(path_X, X)
        np.save(path_ang, angles)
        np.save(path_ids, ids)
        return X, angles, ids


def get_fold_loaders(fold_idx, load_cached_data=True):
    """
    Returns train and validation loaders for a specific fold.
    Performs leak-free imputation of incidence angles.
    """
    set_seed(Config.SEED)

    # 1. Load all training data
    X, angles, y, ids = load_data("train", load_cached_data)

    # 2. Create Stratified K-Fold split
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    # skf.split requires X and y, but X can be zeros as it only uses y for stratification
    splits = list(skf.split(np.zeros(len(y)), y))
    train_idx, val_idx = splits[fold_idx]

    # 3. Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    ang_train, ang_val = angles[train_idx], angles[val_idx]
    ids_train, ids_val = ids[train_idx], ids[val_idx]

    # 4. Leak-Free Imputation of Incidence Angles
    # Calculate median only on training data
    # Note: ang_train may contain NaNs
    valid_angles = ang_train[~np.isnan(ang_train)]
    if len(valid_angles) > 0:
        fill_value = np.median(valid_angles)
    else:
        fill_value = 0.0  # Fallback, though unlikely

    # Fill NaNs
    ang_train_imp = np.copy(ang_train)
    ang_train_imp[np.isnan(ang_train_imp)] = fill_value

    ang_val_imp = np.copy(ang_val)
    ang_val_imp[np.isnan(ang_val_imp)] = fill_value

    # 5. Define Augmentations
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # No TTA or augmentation for validation
    val_transform = None

    # 6. Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train_imp, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val, ang_val_imp, y_val, ids_val, transform=val_transform
    )

    # 7. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm
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
    Returns the test loader.
    Imputes test angles using the median from the full training set.
    """
    set_seed(Config.SEED)

    # 1. Load Test Data
    X_test, ang_test, ids_test = load_data("test", load_cached_data)

    # 2. Load Train Data (for angle imputation statistics)
    _, ang_train, _, _ = load_data("train", load_cached_data)

    # 3. Impute Test Angles
    valid_train_angles = ang_train[~np.isnan(ang_train)]
    if len(valid_train_angles) > 0:
        fill_value = np.median(valid_train_angles)
    else:
        fill_value = 0.0

    ang_test_imp = np.copy(ang_test)
    ang_test_imp[np.isnan(ang_test_imp)] = fill_value

    # 4. Create Dataset
    test_dataset = IcebergDataset(
        X_test, ang_test_imp, labels=None, ids=ids_test, transform=None
    )

    # 5. Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader
