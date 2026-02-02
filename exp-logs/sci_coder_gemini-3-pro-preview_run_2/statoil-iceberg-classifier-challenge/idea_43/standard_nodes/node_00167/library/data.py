import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random

# Import configuration and utilities from the provided library files
from library.config import Config
from library.utils import get_global_stats


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles 3-channel input (Band 1, Band 2, Mean) and incidence angles.
    Implements rotational and flip augmentations.
    """

    def __init__(self, X, inc_angles, y=None, transform=False):
        self.X = X
        self.inc_angles = inc_angles
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is (C, H, W)
        img = self.X[idx]
        inc = self.inc_angles[idx]

        if self.transform:
            # Random Rotation (0, 90, 180, 270 degrees)
            # Image is (C, H, W), so spatial axes are (1, 2)
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k, axes=(1, 2))

            # Random Horizontal Flip
            # Flip along width axis (axis 2)
            if random.random() > 0.5:
                img = np.flip(img, axis=2)

        # Ensure memory is contiguous (fixes negative strides from flips/rotations)
        img = img.copy()

        img_tensor = torch.from_numpy(img).float()
        inc_tensor = torch.tensor([inc], dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor([self.y[idx]], dtype=torch.float32)
            return img_tensor, inc_tensor, label
        else:
            return img_tensor, inc_tensor


def process_data(load_cached_data=True):
    """
    Loads, processes, and normalizes the dataset.
    Uses caching to speed up subsequent runs.

    Normalization Strategy:
    - Band 1 & 2: Min-Max scaling using global stats from library.utils.
    - Band 3 (Mean): Min-Max scaling using stats derived from training data.

    Returns:
        X_train, y_train, inc_train, X_test, inc_test, test_ids
    """
    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(Config.CACHE_FILE):
        print(f"Loading cached data from {Config.CACHE_FILE}...")
        try:
            data = np.load(Config.CACHE_FILE, allow_pickle=True)
            return (
                data["X_train"],
                data["y_train"],
                data["inc_train"],
                data["X_test"],
                data["inc_test"],
                data["test_ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Processing from scratch.")

    print("Processing data from scratch...")

    # 2. Load Raw Data
    df_train = pd.read_json(Config.TRAIN_JSON)
    df_test = pd.read_json(Config.TEST_JSON)

    # 3. Construct Images
    def get_images(df):
        imgs = []
        for _, row in df.iterrows():
            # Reshape flattened bands
            b1 = np.array(row["band_1"]).reshape(75, 75)
            b2 = np.array(row["band_2"]).reshape(75, 75)
            # Band 3: Arithmetic Mean
            b3 = (b1 + b2) / 2.0

            # Stack channels: (75, 75, 3)
            img = np.stack([b1, b2, b3], axis=-1)
            imgs.append(img)
        return np.array(imgs)

    print("Constructing images...")
    X_train = get_images(df_train)
    X_test = get_images(df_test)

    y_train = df_train["is_iceberg"].values.astype(np.float32)
    test_ids = df_test["id"].values

    # 4. Process Incidence Angles (Impute NaNs)
    def process_inc_angle(series):
        vals = pd.to_numeric(series, errors="coerce").values
        mask = ~np.isnan(vals)
        if mask.sum() > 0:
            mean_val = np.mean(vals[mask])
            vals[~mask] = mean_val
        else:
            vals[:] = 0.0  # Fallback
        return vals.astype(np.float32)

    inc_train = process_inc_angle(df_train["inc_angle"])
    inc_test = process_inc_angle(df_test["inc_angle"])

    # 5. Global Normalization
    print("Applying global normalization...")

    # Retrieve global stats for Band 1 and Band 2 using the utility function
    stats = get_global_stats(Config.TRAIN_JSON)

    b1_min, b1_max = stats["band_1"]["min"], stats["band_1"]["max"]
    b2_min, b2_max = stats["band_2"]["min"], stats["band_2"]["max"]

    # Calculate stats for Band 3 (Mean) from the constructed training data
    # X_train is currently (N, 75, 75, 3), so Band 3 is at index 2
    b3_data = X_train[:, :, :, 2]
    b3_min = b3_data.min()
    b3_max = b3_data.max()

    # Create min/max vectors for broadcasting
    min_vals = np.array([b1_min, b2_min, b3_min])
    max_vals = np.array([b1_max, b2_max, b3_max])

    # Apply Scaling: (X - min) / (max - min)
    X_train = (X_train - min_vals) / (max_vals - min_vals)
    X_test = (X_test - min_vals) / (max_vals - min_vals)

    # 6. Transpose to PyTorch Format (N, C, H, W)
    X_train = X_train.transpose(0, 3, 1, 2).astype(np.float32)
    X_test = X_test.transpose(0, 3, 1, 2).astype(np.float32)

    # 7. Save to Cache
    print(f"Saving processed data to {Config.CACHE_FILE}...")
    np.savez(
        Config.CACHE_FILE,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_test=X_test,
        inc_test=inc_test,
        test_ids=test_ids,
    )

    return X_train, y_train, inc_train, X_test, inc_test, test_ids


def make_dataloaders(X, y, inc, train_idx, val_idx, batch_size=Config.BATCH_SIZE):
    """
    Helper function to create Stratified K-Fold DataLoaders.

    Args:
        X, y, inc: Full dataset arrays.
        train_idx, val_idx: Indices for the current fold.
        batch_size: Batch size for loaders.

    Returns:
        train_loader, val_loader
    """
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    inc_tr, inc_val = inc[train_idx], inc[val_idx]

    # Apply augmentation (transform=True) only to training set
    train_ds = IcebergDataset(X_tr, inc_tr, y_tr, transform=True)
    val_ds = IcebergDataset(X_val, inc_val, y_val, transform=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
