import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_PATH,
    WORKING_DIR,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    USE_HORIZONTAL_FLIP,
)
from library.utils import seed_everything

# Ensure reproducibility across the module
seed_everything(SEED)


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75).
            angles (np.ndarray): Shape (N,).
            labels (np.ndarray, optional): Shape (N,). Defaults to None.
            transform (bool): Whether to apply geometric augmentations.
        """
        self.images = torch.tensor(images, dtype=torch.float32)
        self.angles = torch.tensor(angles, dtype=torch.float32)
        self.labels = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        angle = self.angles[idx]

        if self.transform:
            # Random Horizontal Flip
            if USE_HORIZONTAL_FLIP and torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[-1])

            # Random Rotation (0, 90, 180, 270 degrees)
            # k is the number of times to rotate by 90 degrees
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                img = torch.rot90(img, k, dims=[-2, -1])

        # Return format: (inputs, target)
        # Inputs are a tuple of (image, angle)
        if self.labels is not None:
            return (img, angle), self.labels[idx]
        else:
            return (img, angle)


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw data, processes it (3-channel construction, imputation, normalization),
    and caches the result to disk.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    if load_cached_data and os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}")
        data = np.load(CACHE_PATH, allow_pickle=True)
        return (
            data["X_train"],
            data["inc_train"],
            data["y_train"],
            data["X_val"],
            data["inc_val"],
            data["y_val"],
            data["X_test"],
            data["inc_test"],
            data["ids_test"],
        )

    print("Processing data from scratch...")

    # Load Metadata splits
    df_train_meta = pd.read_csv(TRAIN_META_PATH)
    df_val_meta = pd.read_csv(VAL_META_PATH)
    df_test_meta = pd.read_csv(TEST_META_PATH)

    # Load Raw JSON data
    # train.json contains data for both train and val splits
    print(f"Loading {TRAIN_JSON}...")
    with open(TRAIN_JSON, "r") as f:
        train_json_data = json.load(f)
    train_dict = {item["id"]: item for item in train_json_data}

    print(f"Loading {TEST_JSON}...")
    with open(TEST_JSON, "r") as f:
        test_json_data = json.load(f)
    test_dict = {item["id"]: item for item in test_json_data}

    def extract_data(df_meta, source_dict, is_test=False):
        ids = df_meta["id"].values
        # Use incidence angles from metadata (which handles 'na' as NaN)
        inc_angles = df_meta["inc_angle"].values.astype(np.float32)

        images = []
        labels = []

        for img_id in ids:
            item = source_dict[img_id]

            # Reshape flattened bands to 75x75
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            # Construct 3rd channel: Average
            b3 = (b1 + b2) / 2.0

            # Stack to (3, 75, 75)
            img = np.stack([b1, b2, b3], axis=0)
            images.append(img)

            if not is_test:
                labels.append(item["is_iceberg"])

        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.float32) if not is_test else None
        return X, inc_angles, y, ids

    print("Constructing datasets...")
    X_train, inc_train, y_train, _ = extract_data(
        df_train_meta, train_dict, is_test=False
    )
    X_val, inc_val, y_val, _ = extract_data(df_val_meta, train_dict, is_test=False)
    X_test, inc_test, _, ids_test = extract_data(df_test_meta, test_dict, is_test=True)

    # --- Imputation ---
    # Calculate mean incidence angle from TRAIN set only (ignoring NaNs)
    angle_mean = np.nanmean(inc_train)
    print(f"Imputing missing incidence angles with Train mean: {angle_mean:.6f}")

    # Fill NaNs in all sets with the training mean
    inc_train = np.where(np.isnan(inc_train), angle_mean, inc_train)
    inc_val = np.where(np.isnan(inc_val), angle_mean, inc_val)
    inc_test = np.where(np.isnan(inc_test), angle_mean, inc_test)

    # --- Normalization ---
    # Independent Per-Channel Min-Max Scaling
    print("Applying Per-Channel Min-Max Normalization...")
    for c in range(3):
        # Calculate statistics on TRAIN set
        channel_data = X_train[:, c, :, :]
        c_min = channel_data.min()
        c_max = channel_data.max()
        denom = c_max - c_min + 1e-8

        print(f"  Channel {c}: Min={c_min:.4f}, Max={c_max:.4f}")

        # Apply to all sets
        X_train[:, c, :, :] = (X_train[:, c, :, :] - c_min) / denom
        X_val[:, c, :, :] = (X_val[:, c, :, :] - c_min) / denom
        X_test[:, c, :, :] = (X_test[:, c, :, :] - c_min) / denom

    # --- Caching ---
    print(f"Saving processed data to {CACHE_PATH}...")
    np.savez(
        CACHE_PATH,
        X_train=X_train,
        inc_train=inc_train,
        y_train=y_train,
        X_val=X_val,
        inc_val=inc_val,
        y_val=y_val,
        X_test=X_test,
        inc_test=inc_test,
        ids_test=ids_test,
    )

    return (
        X_train,
        inc_train,
        y_train,
        X_val,
        inc_val,
        y_val,
        X_test,
        inc_test,
        ids_test,
    )


def get_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Constructs and returns DataLoaders for Train, Validation, and Test sets.

    Returns:
        train_loader, val_loader, test_loader, ids_test
    """
    (X_train, inc_train, y_train, X_val, inc_val, y_val, X_test, inc_test, ids_test) = (
        process_and_cache_data(load_cached_data)
    )

    # Train Dataset with Augmentation
    train_ds = IcebergDataset(X_train, inc_train, y_train, transform=True)

    # Validation Dataset (No Augmentation)
    val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)

    # Test Dataset (No Augmentation, No Labels)
    test_ds = IcebergDataset(X_test, inc_test, labels=None, transform=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, ids_test
