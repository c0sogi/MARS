import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library import config, utils


# ==========================================
# SCALER
# ==========================================
class GlobalScaler:
    """
    Applies Min-Max scaling per channel based on global statistics.
    Does not clip values, allowing outliers to persist.
    """

    def __init__(self):
        self.mins = None
        self.maxs = None

    def fit(self, X):
        """
        Compute min and max for each channel across the entire dataset X.
        X shape: (N, H, W, C)
        """
        # Reshape to (N * H * W, C) to compute stats per channel
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.mins = X_reshaped.min(axis=0)
        self.maxs = X_reshaped.max(axis=0)
        return self

    def transform(self, X):
        """
        Apply (x - min) / (max - min).
        """
        if self.mins is None or self.maxs is None:
            raise ValueError("Scaler has not been fitted yet.")

        # Broadcast subtraction and division
        # X is (H, W, C) or (N, H, W, C)
        # mins/maxs are (C,)

        numerator = X - self.mins
        denominator = self.maxs - self.mins

        # Avoid division by zero (unlikely with float dB data, but safe)
        denominator = np.where(denominator == 0, 1.0, denominator)

        return numerator / denominator

    def save(self, path):
        np.savez(path, mins=self.mins, maxs=self.maxs)

    def load(self, path):
        data = np.load(path)
        self.mins = data["mins"]
        self.maxs = data["maxs"]


# ==========================================
# DATA PROCESSING & CACHING
# ==========================================
def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes bands into 3-channel images,
    computes global stats, and caches the result.

    Returns:
        data_dict (dict): Contains training and test arrays.
        scaler (GlobalScaler): Fitted scaler.
    """
    logger = utils.get_logger("data_processing")

    cache_path = config.PROCESSED_DATA_CACHE

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached data from {cache_path}")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            data = {
                "X_train": loaded["X_train"],
                "y_train": loaded["y_train"],
                "inc_train": loaded["inc_train"],
                "ids_train": loaded["ids_train"],
                "X_test": loaded["X_test"],
                "inc_test": loaded["inc_test"],
                "ids_test": loaded["ids_test"],
                "scaler_mins": loaded["scaler_mins"],
                "scaler_maxs": loaded["scaler_maxs"],
            }

            scaler = GlobalScaler()
            scaler.mins = data["scaler_mins"]
            scaler.maxs = data["scaler_maxs"]

            return data, scaler
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-processing data.")

    # 2. Process from Scratch
    logger.info("Processing data from source JSONs...")

    # Load Train
    with open(config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)

    # Load Test
    with open(config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # Helper to process list of dicts into arrays
    def process_json_list(data_list, is_train=True):
        ids = []
        bands_1 = []
        bands_2 = []
        inc_angles = []
        labels = []

        for item in data_list:
            ids.append(item["id"])
            bands_1.append(item["band_1"])
            bands_2.append(item["band_2"])
            inc_angles.append(item["inc_angle"])
            if is_train:
                labels.append(item["is_iceberg"])

        # Convert to numpy
        b1 = np.array(bands_1, dtype=np.float32).reshape(-1, 75, 75)
        b2 = np.array(bands_2, dtype=np.float32).reshape(-1, 75, 75)

        # Construct 3rd channel: Mean
        b3 = (b1 + b2) / 2.0

        # Stack: (N, 75, 75, 3)
        X = np.stack([b1, b2, b3], axis=-1)

        # Handle Incidence Angles
        # Convert "na" to NaN, then float
        inc_angles = pd.to_numeric(inc_angles, errors="coerce")
        inc_angles = np.array(inc_angles, dtype=np.float32)

        if is_train:
            y = np.array(labels, dtype=np.float32)
            return X, inc_angles, y, np.array(ids)
        else:
            return X, inc_angles, np.array(ids)

    # Process Train
    X_train, inc_train, y_train, ids_train = process_json_list(
        train_data, is_train=True
    )

    # Process Test
    X_test, inc_test, ids_test = process_json_list(test_data, is_train=False)

    # Impute missing incidence angles with mean of TRAINING set
    inc_mean = np.nanmean(inc_train)

    # Apply imputation
    inc_train = np.where(np.isnan(inc_train), inc_mean, inc_train)
    # Note: Test set might also have missing values or "na"
    inc_test = np.where(np.isnan(inc_test), inc_mean, inc_test)

    # Fit Global Scaler on Training Data
    scaler = GlobalScaler()
    scaler.fit(X_train)

    # Save to Cache
    logger.info(f"Saving processed data to {cache_path}")
    np.savez(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        ids_train=ids_train,
        X_test=X_test,
        inc_test=inc_test,
        ids_test=ids_test,
        scaler_mins=scaler.mins,
        scaler_maxs=scaler.maxs,
    )

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "inc_train": inc_train,
        "ids_train": ids_train,
        "X_test": X_test,
        "inc_test": inc_test,
        "ids_test": ids_test,
    }

    return data, scaler


# ==========================================
# DATASET CLASS
# ==========================================
class IcebergDataset(Dataset):
    def __init__(self, X, inc_angles, labels=None, transform=None, scaler=None):
        """
        Args:
            X: (N, 75, 75, 3) numpy array
            inc_angles: (N,) numpy array
            labels: (N,) numpy array or None
            transform: Albumentations transform
            scaler: GlobalScaler instance
        """
        self.X = X
        self.inc_angles = inc_angles
        self.labels = labels
        self.transform = transform
        self.scaler = scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get image
        img = self.X[idx]  # (75, 75, 3)
        angle = self.inc_angles[idx]

        # Apply Scaling
        if self.scaler:
            img = self.scaler.transform(img)

        # Ensure float32
        img = img.astype(np.float32)
        angle = np.array([angle], dtype=np.float32)  # Wrap in array for batching

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            # Default to tensor conversion if no transform provided
            # Albumentations ToTensorV2 handles HWC -> CHW
            converter = ToTensorV2()
            img = converter(image=img)["image"]

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, torch.tensor(angle), label
        else:
            return img, torch.tensor(angle)


# ==========================================
# AUGMENTATIONS
# ==========================================
def get_transforms(phase="train"):
    if phase == "train":
        return A.Compose(
            [
                # Rotational Invariance: 0, 90, 180, 270
                A.RandomRotate90(p=0.5),
                # Horizontal Flip
                A.HorizontalFlip(p=0.5),
                # No Vertical Flip, No Mixup as per instructions
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# ==========================================
# DATALOADERS
# ==========================================
def get_dataloaders(fold_idx, data, scaler):
    """
    Creates dataloaders for a specific fold using StratifiedKFold.

    Args:
        fold_idx (int): The current fold index (0 to NUM_FOLDS-1).
        data (dict): The dictionary returned by process_and_cache_data.
        scaler (GlobalScaler): The fitted scaler.

    Returns:
        train_loader, val_loader, test_loader
    """
    X_all = data["X_train"]
    y_all = data["y_train"]
    inc_all = data["inc_train"]

    X_test = data["X_test"]
    inc_test = data["inc_test"]

    # Debugging: Subsample if configured
    if config.DEBUG_MAX_SAMPLES is not None:
        limit = min(len(X_all), config.DEBUG_MAX_SAMPLES)
        X_all = X_all[:limit]
        y_all = y_all[:limit]
        inc_all = inc_all[:limit]

        limit_test = min(len(X_test), config.DEBUG_MAX_SAMPLES)
        X_test = X_test[:limit_test]
        inc_test = inc_test[:limit_test]

    # Stratified Split
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Get indices for the requested fold
    # We iterate to find the fold_idx-th split
    splits = list(skf.split(X_all, y_all))
    train_idx, val_idx = splits[fold_idx]

    # Create subsets
    X_train_fold = X_all[train_idx]
    y_train_fold = y_all[train_idx]
    inc_train_fold = inc_all[train_idx]

    X_val_fold = X_all[val_idx]
    y_val_fold = y_all[val_idx]
    inc_val_fold = inc_all[val_idx]

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train_fold,
        inc_train_fold,
        y_train_fold,
        transform=get_transforms("train"),
        scaler=scaler,
    )

    val_dataset = IcebergDataset(
        X_val_fold,
        inc_val_fold,
        y_val_fold,
        transform=get_transforms("val"),
        scaler=scaler,
    )

    test_dataset = IcebergDataset(
        X_test, inc_test, labels=None, transform=get_transforms("test"), scaler=scaler
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
