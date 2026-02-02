import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles on-the-fly augmentation using native tensor operations.
    """

    def __init__(self, images, angles, labels=None, augment=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75), pre-normalized.
            angles (np.ndarray): Shape (N,), imputed incidence angles.
            labels (np.ndarray, optional): Shape (N,), target labels.
            augment (bool): Whether to apply geometric augmentations.
        """
        # Convert to float tensors
        self.images = torch.FloatTensor(images)
        self.angles = torch.FloatTensor(angles)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        angle = self.angles[idx]

        # Apply Augmentations (Training Only)
        if self.augment:
            # Random Rotation: 0, 90, 180, 270 degrees
            # k is number of times to rotate by 90 degrees
            k = np.random.randint(0, 4)
            img = torch.rot90(img, k, dims=[1, 2])

            # Random Horizontal Flip (p=0.5)
            # Flip along width axis (dim 2)
            if np.random.random() > 0.5:
                img = torch.flip(img, dims=[2])

            # Note: Vertical Flips excluded as per instructions.

        if self.labels is not None:
            return img, angle, self.labels[idx]
        else:
            return img, angle


def load_and_process_data(load_cached_data=True):
    """
    Loads raw JSON data, processes into 3-channel tensors, and handles caching.

    Processing:
    - Band 1, Band 2, Mean(Band 1, Band 2)
    - Reshape to (N, 3, 75, 75)
    - Parse inc_angle (handling 'na')

    Returns:
        tuple: (train_images, train_angles, train_labels, train_ids,
                test_images, test_angles, test_ids)
        All returned arrays are raw (unscaled) numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(Config.CACHE_PATH):
        print(f"Loading cached data from {Config.CACHE_PATH}")
        try:
            data = np.load(Config.CACHE_PATH)
            return (
                data["train_images"],
                data["train_angles"],
                data["train_labels"],
                data["train_ids"],
                data["test_images"],
                data["test_angles"],
                data["test_ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data from scratch...")

    # 2. Process from Scratch
    print("Processing data from scratch...")

    def process_json(file_path, is_train=True):
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        ids = []
        bands_1 = []
        bands_2 = []
        angles = []
        labels = []

        for item in raw_data:
            ids.append(item["id"])
            bands_1.append(item["band_1"])
            bands_2.append(item["band_2"])
            angles.append(item["inc_angle"])
            if is_train:
                labels.append(item["is_iceberg"])

        # Convert bands to numpy arrays and reshape
        # Raw data is flattened 75x75 list
        b1 = np.array(bands_1, dtype=np.float32).reshape(-1, 75, 75)
        b2 = np.array(bands_2, dtype=np.float32).reshape(-1, 75, 75)

        # Construct 3rd Channel: Arithmetic Mean
        b3 = (b1 + b2) / 2.0

        # Stack to (N, 3, 75, 75)
        images = np.stack([b1, b2, b3], axis=1)

        # Process Incidence Angles
        # Coerce 'na' to NaN
        angles = pd.to_numeric(angles, errors="coerce").astype(np.float32)
        angles = np.array(angles)

        ids = np.array(ids)

        if is_train:
            labels = np.array(labels, dtype=np.float32)
            return images, angles, labels, ids
        else:
            return images, angles, ids

    # Process Train and Test
    train_images, train_angles, train_labels, train_ids = process_json(
        Config.TRAIN_JSON, is_train=True
    )
    test_images, test_angles, test_ids = process_json(Config.TEST_JSON, is_train=False)

    # Handle DEBUG mode
    if Config.DEBUG:
        print(f"DEBUG Mode: Truncating datasets to {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_images = train_images[: Config.DEBUG_SAMPLE_SIZE]
        train_angles = train_angles[: Config.DEBUG_SAMPLE_SIZE]
        train_labels = train_labels[: Config.DEBUG_SAMPLE_SIZE]
        train_ids = train_ids[: Config.DEBUG_SAMPLE_SIZE]

        test_images = test_images[: Config.DEBUG_SAMPLE_SIZE]
        test_angles = test_angles[: Config.DEBUG_SAMPLE_SIZE]
        test_ids = test_ids[: Config.DEBUG_SAMPLE_SIZE]

    # 3. Save to Cache
    print(f"Saving processed data to {Config.CACHE_PATH}")
    np.savez(
        Config.CACHE_PATH,
        train_images=train_images,
        train_angles=train_angles,
        train_labels=train_labels,
        train_ids=train_ids,
        test_images=test_images,
        test_angles=test_angles,
        test_ids=test_ids,
    )

    return (
        train_images,
        train_angles,
        train_labels,
        train_ids,
        test_images,
        test_angles,
        test_ids,
    )


def get_fold_dataloaders(fold_idx, load_cached_data=True):
    """
    Generates DataLoaders for a specific fold using Stratified K-Fold.

    Strict Fold-wise Preprocessing:
    - Calculates Min/Max scaling stats ONLY on the training split.
    - Calculates Angle Mean imputation ONLY on the training split.
    - Applies these stats to the validation split.

    Returns:
        train_loader, val_loader, stats (dict)
    """
    # Load raw data
    train_imgs, train_angles, train_labels, train_ids, _, _, _ = load_and_process_data(
        load_cached_data
    )

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    splits = list(skf.split(train_imgs, train_labels))
    if fold_idx >= len(splits):
        raise ValueError(f"Fold index {fold_idx} out of range.")

    train_idx, val_idx = splits[fold_idx]

    # Split Data
    X_train, X_val = train_imgs[train_idx], train_imgs[val_idx]
    y_train, y_val = train_labels[train_idx], train_labels[val_idx]
    ang_train, ang_val = train_angles[train_idx], train_angles[val_idx]

    # --- Strict Fold-wise Preprocessing ---

    # 1. Image Normalization (Independent Per-Channel Min-Max)
    # Compute stats on training set only
    # Shape: (N, 3, 75, 75) -> Min/Max over (N, H, W) -> (1, 3, 1, 1)
    min_vals = X_train.min(axis=(0, 2, 3), keepdims=True)
    max_vals = X_train.max(axis=(0, 2, 3), keepdims=True)

    # Apply scaling
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0  # Prevent division by zero

    X_train = (X_train - min_vals) / denom
    X_val = (X_val - min_vals) / denom

    # 2. Angle Imputation
    # Compute mean on training set only (ignoring NaNs)
    angle_mean = np.nanmean(ang_train)
    if np.isnan(angle_mean):
        angle_mean = 0.0

    # Fill NaNs
    ang_train = np.where(np.isnan(ang_train), angle_mean, ang_train)
    ang_val = np.where(np.isnan(ang_val), angle_mean, ang_val)

    # Create Datasets
    # Augmentation enabled for Train, disabled for Val
    train_dataset = IcebergDataset(X_train, ang_train, y_train, augment=True)
    val_dataset = IcebergDataset(X_val, ang_val, y_val, augment=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Bundle stats for use in testing
    stats = {"min_vals": min_vals, "max_vals": max_vals, "angle_mean": angle_mean}

    return train_loader, val_loader, stats


def get_test_dataloader(stats, load_cached_data=True):
    """
    Generates the Test DataLoader.

    Crucial: Applies the normalization statistics (min/max/angle_mean)
    derived from a specific training fold to the test set.
    """
    # Load raw test data
    _, _, _, _, test_imgs, test_angles, test_ids = load_and_process_data(
        load_cached_data
    )

    # Unpack stats
    min_vals = stats["min_vals"]
    max_vals = stats["max_vals"]
    angle_mean = stats["angle_mean"]

    # Apply Normalization
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0
    test_imgs = (test_imgs - min_vals) / denom

    # Apply Imputation
    test_angles = np.where(np.isnan(test_angles), angle_mean, test_angles)

    # Create Dataset (No Augmentation)
    test_dataset = IcebergDataset(test_imgs, test_angles, labels=None, augment=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_loader, test_ids
