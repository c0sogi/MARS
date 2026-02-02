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
    Handles 3-channel image construction and incidence angle integration.
    """

    def __init__(self, X, angles, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Image data of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Data is already (C, H, W) float32
        img = torch.from_numpy(self.X[idx])
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, angle, label
        else:
            # For test set, we might want IDs, but the loader logic usually handles IDs separately
            # or we just return img/angle. The prompt implies predicting for IDs.
            # To keep it simple and consistent with typical loops, we return img, angle.
            # IDs are managed by the caller via the order of the dataset.
            return img, angle


def _process_json_data(json_path, is_train=True):
    """
    Reads JSON, processes bands into (3, 75, 75) images, handles angles, and extracts labels.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data)

    # Process Images
    # Band 1 (HH)
    b1 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]]
    )
    # Band 2 (HV)
    b2 = np.array(
        [np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]]
    )
    # Band 3 (Average)
    b3 = (b1 + b2) / 2.0

    # Stack to (N, 3, 75, 75) - PyTorch channel first format
    # np.stack creates (N, 75, 75, 3) if axis=-1, or (N, 3, 75, 75) if axis=1.
    # Let's stack along axis 1.
    X = np.stack([b1, b2, b3], axis=1)

    # Process Angles
    # Replace 'na' with NaN, then fill with impute value
    angles = pd.to_numeric(df["inc_angle"], errors="coerce")
    angles = angles.fillna(Config.INC_ANGLE_IMPUTE_VAL).values.astype(np.float32)

    # Process IDs
    ids = df["id"].values

    if is_train:
        y = df["is_iceberg"].values.astype(np.float32)
        return X, angles, y, ids
    else:
        return X, angles, ids


def _get_cache_paths(prefix):
    return {
        "X": os.path.join(Config.WORKING_DIR, f"X_{prefix}.npy"),
        "angles": os.path.join(Config.WORKING_DIR, f"angles_{prefix}.npy"),
        "y": os.path.join(Config.WORKING_DIR, f"y_{prefix}.npy"),
        "ids": os.path.join(Config.WORKING_DIR, f"ids_{prefix}.npy"),
    }


def load_processed_data(is_train=True, load_cached_data=True):
    """
    Loads data from cache if available and requested, otherwise processes from JSON and caches it.
    """
    prefix = "train" if is_train else "test"
    paths = _get_cache_paths(prefix)

    # Check if cache exists
    cache_exists = (
        os.path.exists(paths["X"])
        and os.path.exists(paths["angles"])
        and os.path.exists(paths["ids"])
    )
    if is_train:
        cache_exists = cache_exists and os.path.exists(paths["y"])

    if load_cached_data and cache_exists:
        # Load from cache
        X = np.load(paths["X"])
        angles = np.load(paths["angles"])
        ids = np.load(paths["ids"], allow_pickle=True)
        y = np.load(paths["y"]) if is_train else None
    else:
        # Process from scratch
        json_path = Config.TRAIN_JSON if is_train else Config.TEST_JSON
        if is_train:
            X, angles, y, ids = _process_json_data(json_path, is_train=True)
            np.save(paths["y"], y)
        else:
            X, angles, ids = _process_json_data(json_path, is_train=False)
            y = None

        np.save(paths["X"], X)
        np.save(paths["angles"], angles)
        np.save(paths["ids"], ids)

    return X, angles, y, ids


def get_train_val_loaders(fold_index, load_cached_data=True):
    """
    Returns train and validation DataLoaders for a specific fold.

    Args:
        fold_index (int): Index of the fold (0 to N_FOLDS-1).
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        train_loader, val_loader
    """
    # Load full training data
    X, angles, y, ids = load_processed_data(
        is_train=True, load_cached_data=load_cached_data
    )

    # Debugging: Subset if MAX_SAMPLES is set
    if Config.MAX_SAMPLES is not None:
        X = X[: Config.MAX_SAMPLES]
        angles = angles[: Config.MAX_SAMPLES]
        y = y[: Config.MAX_SAMPLES]
        ids = ids[: Config.MAX_SAMPLES]

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Generate indices
    splits = list(skf.split(X, y))
    if fold_index >= len(splits):
        raise ValueError(
            f"Fold index {fold_index} out of range for {Config.N_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_index]

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    angles_train, angles_val = angles[train_idx], angles[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Define Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angles_val, y_val, transform=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
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
    Returns the test DataLoader and the list of test IDs.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        test_loader, test_ids
    """
    # Load full test data
    X, angles, _, ids = load_processed_data(
        is_train=False, load_cached_data=load_cached_data
    )

    # Create Dataset (No augmentation for test)
    test_dataset = IcebergDataset(X, angles, y=None, transform=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return test_loader, ids
