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
    def __init__(self, images, angles, targets=None, ids=None, transform=None):
        """
        PyTorch Dataset for Iceberg vs Ship classification.

        Args:
            images (np.ndarray): Image data of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            targets (np.ndarray, optional): Binary targets of shape (N,).
            ids (np.ndarray, optional): Image IDs of shape (N,).
            transform (callable, optional): Torchvision transforms for augmentation.
        """
        self.images = images
        self.angles = angles
        self.targets = targets
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.images[idx]
        angle = self.angles[idx]

        # Convert image to FloatTensor
        # Input is (3, 75, 75), float32
        img_tensor = torch.from_numpy(img).float()

        # Apply augmentations (if any)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Convert angle to FloatTensor
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.targets is not None:
            # Training/Validation mode
            label = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
            return img_tensor, angle_tensor, label
        else:
            # Inference mode
            img_id = self.ids[idx]
            return img_tensor, angle_tensor, img_id


def _process_split(metadata_path, raw_data_dict, angle_imputer_val=None):
    """
    Internal helper to process a specific data split (train/val/test) from raw JSON.
    """
    df = pd.read_csv(metadata_path)

    # Filter raw data using original_index to avoid O(N^2) lookups
    indices = df["original_index"].values
    subset = [raw_data_dict[i] for i in indices]

    # Process Band Data
    # Band 1: HH
    b1 = np.array([x["band_1"] for x in subset], dtype=np.float32).reshape(-1, 75, 75)
    # Band 2: HV
    b2 = np.array([x["band_2"] for x in subset], dtype=np.float32).reshape(-1, 75, 75)
    # Band 3: Synthetic Average ((HH + HV) / 2)
    b3 = (b1 + b2) / 2.0

    # Stack channels: (N, 3, 75, 75)
    X = np.stack([b1, b2, b3], axis=1)

    # Process Incidence Angles
    # Use metadata column which preserves 'na' as NaN
    angles = df["inc_angle"].values.astype(np.float32)

    # Impute missing angles
    if angle_imputer_val is not None:
        mask = np.isnan(angles)
        angles[mask] = angle_imputer_val

    # Process Targets and IDs
    ids = df["id"].values.astype(str)
    y = None
    if "is_iceberg" in df.columns:
        y = df["is_iceberg"].values.astype(np.float32)

    return X, angles, y, ids


def get_data(config: Config, load_cached_data: bool = True):
    """
    Retrieves dataset arrays, utilizing a caching mechanism to avoid re-processing JSONs.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays for train, val, and test sets.
    """
    # Define cache file paths
    cache_files = {
        "X_train": "X_train.npy",
        "angle_train": "angle_train.npy",
        "y_train": "y_train.npy",
        "ids_train": "ids_train.npy",
        "X_val": "X_val.npy",
        "angle_val": "angle_val.npy",
        "y_val": "y_val.npy",
        "ids_val": "ids_val.npy",
        "X_test": "X_test.npy",
        "angle_test": "angle_test.npy",
        "ids_test": "ids_test.npy",
    }

    # Check if all cache files exist
    cache_exists = True
    for fname in cache_files.values():
        if not os.path.exists(config.get_cache_path(fname)):
            cache_exists = False
            break

    # Load from cache if requested and available
    if load_cached_data and cache_exists:
        data = {}
        for key, fname in cache_files.items():
            # allow_pickle=True needed for string arrays (ids), safe for trusted local files
            data[key] = np.load(
                config.get_cache_path(fname), allow_pickle=("ids" in key)
            )
        return data

    # Process from scratch
    # Load raw JSON files (Memory intensive but fits in provided environment)
    with open(config.train_json_path, "r") as f:
        raw_train = json.load(f)
    with open(config.test_json_path, "r") as f:
        raw_test = json.load(f)

    # Determine imputation value for incidence angle
    # Strictly use training set median to prevent leakage
    train_meta = pd.read_csv(config.train_meta_path)
    train_angles = pd.to_numeric(train_meta["inc_angle"], errors="coerce")
    angle_median = train_angles.median()

    # Process each split
    X_train, ang_train, y_train, ids_train = _process_split(
        config.train_meta_path, raw_train, angle_median
    )
    X_val, ang_val, y_val, ids_val = _process_split(
        config.val_meta_path, raw_train, angle_median
    )
    X_test, ang_test, _, ids_test = _process_split(
        config.test_meta_path, raw_test, angle_median
    )

    # Save to cache
    np.save(config.get_cache_path("X_train.npy"), X_train)
    np.save(config.get_cache_path("angle_train.npy"), ang_train)
    np.save(config.get_cache_path("y_train.npy"), y_train)
    np.save(config.get_cache_path("ids_train.npy"), ids_train)

    np.save(config.get_cache_path("X_val.npy"), X_val)
    np.save(config.get_cache_path("angle_val.npy"), ang_val)
    np.save(config.get_cache_path("y_val.npy"), y_val)
    np.save(config.get_cache_path("ids_val.npy"), ids_val)

    np.save(config.get_cache_path("X_test.npy"), X_test)
    np.save(config.get_cache_path("angle_test.npy"), ang_test)
    np.save(config.get_cache_path("ids_test.npy"), ids_test)

    data = {
        "X_train": X_train,
        "angle_train": ang_train,
        "y_train": y_train,
        "ids_train": ids_train,
        "X_val": X_val,
        "angle_val": ang_val,
        "y_val": y_val,
        "ids_val": ids_val,
        "X_test": X_test,
        "angle_test": ang_test,
        "ids_test": ids_test,
    }

    return data


def get_loaders(config: Config, fold_idx: int = None):
    """
    Constructs DataLoaders for training, validation, and testing.

    Args:
        config (Config): Configuration object.
        fold_idx (int, optional): The fold index (0-4) for Cross-Validation.
                                  If None, uses the static train/val split from metadata.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Retrieve data (cached if available)
    data = get_data(config, load_cached_data=True)

    # Determine Train/Val Split strategy
    if fold_idx is not None:
        # 5-Fold Cross Validation Strategy:
        # Merge static train and val sets, then split dynamically.
        X_all = np.concatenate([data["X_train"], data["X_val"]], axis=0)
        angle_all = np.concatenate([data["angle_train"], data["angle_val"]], axis=0)
        y_all = np.concatenate([data["y_train"], data["y_val"]], axis=0)

        skf = StratifiedKFold(
            n_splits=config.n_folds, shuffle=True, random_state=config.seed
        )
        splits = list(skf.split(X_all, y_all))

        if fold_idx >= len(splits):
            raise ValueError(
                f"Fold index {fold_idx} out of range for {config.n_folds} folds."
            )

        train_idx, val_idx = splits[fold_idx]

        X_train_fold = X_all[train_idx]
        angle_train_fold = angle_all[train_idx]
        y_train_fold = y_all[train_idx]

        X_val_fold = X_all[val_idx]
        angle_val_fold = angle_all[val_idx]
        y_val_fold = y_all[val_idx]
    else:
        # Static Split Strategy (Default/Debug):
        X_train_fold = data["X_train"]
        angle_train_fold = data["angle_train"]
        y_train_fold = data["y_train"]

        X_val_fold = data["X_val"]
        angle_val_fold = data["angle_val"]
        y_val_fold = data["y_val"]

    # Test Data
    X_test = data["X_test"]
    angle_test = data["angle_test"]
    ids_test = data["ids_test"]

    # Debug Subsetting
    if config.debug:
        limit = 32
        X_train_fold = X_train_fold[:limit]
        angle_train_fold = angle_train_fold[:limit]
        y_train_fold = y_train_fold[:limit]

        X_val_fold = X_val_fold[:limit]
        angle_val_fold = angle_val_fold[:limit]
        y_val_fold = y_val_fold[:limit]

        X_test = X_test[:limit]
        angle_test = angle_test[:limit]
        ids_test = ids_test[:limit]

    # Define Augmentations
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Instantiate Datasets
    train_ds = IcebergDataset(
        X_train_fold, angle_train_fold, y_train_fold, transform=train_transform
    )
    val_ds = IcebergDataset(X_val_fold, angle_val_fold, y_val_fold)
    test_ds = IcebergDataset(X_test, angle_test, ids=ids_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
