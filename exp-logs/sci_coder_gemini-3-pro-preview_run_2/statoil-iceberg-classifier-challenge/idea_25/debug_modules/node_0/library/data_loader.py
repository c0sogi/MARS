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
    Custom Dataset for Iceberg/Ship classification.
    Handles on-the-fly scaling and augmentation.
    """

    def __init__(self, images, inc_angles, labels=None, scalers=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 75, 75, 3)
            inc_angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,). Defaults to None.
            scalers (dict, optional): {'min': [c0, c1, c2], 'max': [c0, c1, c2]}.
            transform (bool): Whether to apply augmentation.
        """
        self.images = images
        self.inc_angles = inc_angles
        self.labels = labels
        self.scalers = scalers
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (H, W, C)
        image = self.images[idx].copy()
        inc_angle = self.inc_angles[idx]

        # 1. Apply Scaling (Min-Max)
        if self.scalers is not None:
            min_vals = np.array(self.scalers["min"])
            max_vals = np.array(self.scalers["max"])
            # Avoid division by zero
            denom = max_vals - min_vals
            denom[denom == 0] = 1.0
            image = (image - min_vals) / denom

        # 2. Apply Augmentation (Training only)
        if self.transform:
            # Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            image = np.rot90(image, k=k)

            # Random Horizontal Flip
            if np.random.random() < 0.5:
                image = np.fliplr(image)

            # Note: Vertical flip is excluded per requirements

        # 3. Convert to Tensor (C, H, W)
        # Current shape is (75, 75, 3), PyTorch expects (3, 75, 75)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image).float()

        # 4. Handle Label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image_tensor, torch.tensor(inc_angle, dtype=torch.float32), label
        else:
            return image_tensor, torch.tensor(inc_angle, dtype=torch.float32)


def process_json_data(json_path, target_ids=None):
    """
    Helper to load JSON and process bands into numpy arrays.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Filter by IDs if provided
    if target_ids is not None:
        target_set = set(target_ids)
        data = [item for item in data if item["id"] in target_set]
        # Sort to ensure alignment with metadata if needed,
        # but we usually rely on the order of the list or re-index later.
        # Here we just keep the list order.

    # Pre-allocate arrays
    count = len(data)
    images = np.zeros((count, 75, 75, 3), dtype=np.float32)
    inc_angles = []
    ids = []
    labels = []
    has_labels = "is_iceberg" in data[0]

    for i, item in enumerate(data):
        # Process Bands
        band_1 = np.array(item["band_1"]).reshape(75, 75)
        band_2 = np.array(item["band_2"]).reshape(75, 75)
        band_avg = (band_1 + band_2) / 2.0

        images[i, :, :, 0] = band_1
        images[i, :, :, 1] = band_2
        images[i, :, :, 2] = band_avg

        # Process Inc Angle
        angle = item["inc_angle"]
        if angle == "na":
            inc_angles.append(np.nan)
        else:
            inc_angles.append(float(angle))

        ids.append(item["id"])

        if has_labels:
            labels.append(item["is_iceberg"])

    inc_angles = np.array(inc_angles, dtype=np.float32)
    if has_labels:
        labels = np.array(
            labels, dtype=np.int64
        )  # Long for classification if using CrossEntropy, Float for BCE
    else:
        labels = None

    return images, inc_angles, labels, np.array(ids)


def load_data(debug=False, load_cached_data=True):
    """
    Loads data from JSON or Cache.
    Combines train and val metadata splits for full cross-validation.
    """
    os.makedirs(os.path.dirname(Config.CACHE_PATH), exist_ok=True)

    if load_cached_data and os.path.exists(Config.CACHE_PATH):
        print(f"Loading cached data from {Config.CACHE_PATH}...")
        try:
            data = np.load(Config.CACHE_PATH, allow_pickle=True)
            X_train = data["X_train"]
            inc_train = data["inc_train"]
            y_train = data["y_train"]
            train_ids = data["train_ids"]
            X_test = data["X_test"]
            inc_test = data["inc_test"]
            test_ids = data["test_ids"]

            if debug:
                return (
                    X_train[: Config.DEBUG_SIZE],
                    inc_train[: Config.DEBUG_SIZE],
                    y_train[: Config.DEBUG_SIZE],
                    train_ids[: Config.DEBUG_SIZE],
                ), (
                    X_test[: Config.DEBUG_SIZE],
                    inc_test[: Config.DEBUG_SIZE],
                    test_ids[: Config.DEBUG_SIZE],
                )

            return (X_train, inc_train, y_train, train_ids), (
                X_test,
                inc_test,
                test_ids,
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Processing from scratch.")

    print("Processing raw data from JSON...")

    # 1. Identify Training IDs (Combine Train + Val from metadata)
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    df_val_meta = pd.read_csv(Config.VAL_META)

    # Combine IDs to get full training set for CV
    full_train_ids = pd.concat([df_train_meta["id"], df_val_meta["id"]]).unique()

    # 2. Identify Test IDs
    df_test_meta = pd.read_csv(Config.TEST_META)
    test_ids_target = df_test_meta["id"].unique()

    # 3. Process Train Data
    X_train, inc_train, y_train, train_ids = process_json_data(
        Config.TRAIN_JSON, full_train_ids
    )

    # 4. Process Test Data
    X_test, inc_test, _, test_ids = process_json_data(Config.TEST_JSON, test_ids_target)

    # 5. Save Cache
    np.savez(
        Config.CACHE_PATH,
        X_train=X_train,
        inc_train=inc_train,
        y_train=y_train,
        train_ids=train_ids,
        X_test=X_test,
        inc_test=inc_test,
        test_ids=test_ids,
    )
    print(f"Data processed and saved to {Config.CACHE_PATH}")

    if debug:
        return (
            X_train[: Config.DEBUG_SIZE],
            inc_train[: Config.DEBUG_SIZE],
            y_train[: Config.DEBUG_SIZE],
            train_ids[: Config.DEBUG_SIZE],
        ), (
            X_test[: Config.DEBUG_SIZE],
            inc_test[: Config.DEBUG_SIZE],
            test_ids[: Config.DEBUG_SIZE],
        )

    return (X_train, inc_train, y_train, train_ids), (X_test, inc_test, test_ids)


def get_dataloaders(
    fold_index, train_data, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates train and validation dataloaders for a specific fold.
    Performs strict fold-wise scaling and imputation.
    """
    X, inc, y, ids = train_data

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the specific fold
    # We iterate to find the Nth fold
    splits = list(skf.split(X, y))
    if fold_index >= len(splits):
        raise ValueError(
            f"Fold index {fold_index} out of range for {Config.NUM_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_index]

    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    inc_train, inc_val = inc[train_idx], inc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # --- Strict Preprocessing (Fit on Train, Apply to Val) ---

    # 1. Impute inc_angle
    # Calculate mean on training set (ignoring NaNs)
    inc_mean = np.nanmean(inc_train)

    # Fill NaNs
    inc_train_filled = np.where(np.isnan(inc_train), inc_mean, inc_train)
    inc_val_filled = np.where(np.isnan(inc_val), inc_mean, inc_val)

    # 2. Calculate Scalers (Min-Max per channel)
    # X shape: (N, 75, 75, 3)
    # We want min/max for each of the 3 channels across N, H, W
    scalers = {"min": [], "max": []}
    for c in range(3):
        channel_data = X_train[:, :, :, c]
        c_min = np.min(channel_data)
        c_max = np.max(channel_data)
        scalers["min"].append(c_min)
        scalers["max"].append(c_max)

    # --- Create Datasets ---

    train_dataset = IcebergDataset(
        images=X_train,
        inc_angles=inc_train_filled,
        labels=y_train,
        scalers=scalers,
        transform=True,  # Apply augmentation to training
    )

    val_dataset = IcebergDataset(
        images=X_val,
        inc_angles=inc_val_filled,
        labels=y_val,
        scalers=scalers,
        transform=False,  # No augmentation for validation
    )

    # --- Create Loaders ---

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader


def get_test_loader(
    train_data, test_data, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates test dataloader.
    Uses statistics derived from the FULL training set.
    """
    X_train, inc_train, _, _ = train_data
    X_test, inc_test, test_ids = test_data

    # 1. Impute inc_angle using full training mean
    inc_mean = np.nanmean(inc_train)
    inc_test_filled = np.where(np.isnan(inc_test), inc_mean, inc_test)

    # 2. Calculate Scalers on full training set
    scalers = {"min": [], "max": []}
    for c in range(3):
        channel_data = X_train[:, :, :, c]
        c_min = np.min(channel_data)
        c_max = np.max(channel_data)
        scalers["min"].append(c_min)
        scalers["max"].append(c_max)

    # 3. Create Dataset
    test_dataset = IcebergDataset(
        images=X_test,
        inc_angles=inc_test_filled,
        labels=None,
        scalers=scalers,
        transform=False,
    )

    # 4. Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return test_loader, test_ids
