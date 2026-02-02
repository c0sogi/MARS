import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config


def get_data(load_cached_data=True):
    """
    Loads training and test data from JSON files.
    Implements caching mechanism using .npz files to optimize runtime.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_data (dict): Contains 'images', 'labels', 'angles', 'ids'.
        test_data (dict): Contains 'images', 'angles', 'ids'.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "processed_data.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            train_data = {
                "images": data["train_images"],
                "labels": data["train_labels"],
                "angles": data["train_angles"],
                "ids": data["train_ids"],
            }
            test_data = {
                "images": data["test_images"],
                "angles": data["test_angles"],
                "ids": data["test_ids"],
            }
            return train_data, test_data
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Process from Scratch
    print("Processing data from raw JSON files...")

    # --- Process Training Data ---
    with open(Config.TRAIN_JSON, "r") as f:
        train_json = json.load(f)
    df_train = pd.DataFrame(train_json)

    # Process Images: Convert lists to (N, 75, 75, 2) array
    # Band 1 and Band 2 are flattened lists of 5625 floats
    x_band1 = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_1"]
        ]
    )
    x_band2 = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_2"]
        ]
    )
    train_images = np.stack([x_band1, x_band2], axis=-1)

    # Process Angles: Handle 'na' values
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    inc_angle_mean = df_train["inc_angle"].mean()
    df_train["inc_angle"] = df_train["inc_angle"].fillna(inc_angle_mean)
    train_angles = df_train["inc_angle"].values.astype(np.float32)

    train_labels = df_train["is_iceberg"].values.astype(np.int64)
    train_ids = df_train["id"].values

    # --- Process Test Data ---
    with open(Config.TEST_JSON, "r") as f:
        test_json = json.load(f)
    df_test = pd.DataFrame(test_json)

    x_band1_test = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_test["band_1"]
        ]
    )
    x_band2_test = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_test["band_2"]
        ]
    )
    test_images = np.stack([x_band1_test, x_band2_test], axis=-1)

    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
    # Fill missing test angles with training mean
    df_test["inc_angle"] = df_test["inc_angle"].fillna(inc_angle_mean)
    test_angles = df_test["inc_angle"].values.astype(np.float32)

    test_ids = df_test["id"].values

    # 3. Save to Cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez(
        cache_path,
        train_images=train_images,
        train_labels=train_labels,
        train_angles=train_angles,
        train_ids=train_ids,
        test_images=test_images,
        test_angles=test_angles,
        test_ids=test_ids,
    )
    print(f"Data processed and cached to {cache_path}")

    train_data = {
        "images": train_images,
        "labels": train_labels,
        "angles": train_angles,
        "ids": train_ids,
    }
    test_data = {"images": test_images, "angles": test_angles, "ids": test_ids}

    return train_data, test_data


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=False, scaler_stats=None):
        """
        Custom Dataset for Iceberg/Ship classification.

        Args:
            images: (N, 75, 75, 2) numpy array
            angles: (N,) numpy array
            labels: (N,) numpy array or None
            transform: bool, whether to apply augmentations
            scaler_stats: tuple (min_vals, max_vals) for 3 channels.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform
        self.scaler_stats = scaler_stats

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve raw data (75, 75, 2)
        img = self.images[idx]
        angle = self.angles[idx]

        # 1. Construct 3rd Channel (Average)
        # (Band1 + Band2) / 2
        avg_channel = (img[..., 0] + img[..., 1]) / 2.0
        # Stack to (75, 75, 3)
        img_3ch = np.dstack((img, avg_channel))

        # 2. Independent Per-Channel Min-Max Scaling
        if self.scaler_stats is not None:
            min_vals, max_vals = self.scaler_stats
            # min_vals/max_vals are shape (3,)
            # Broadcast over (75, 75, 3)
            denom = max_vals - min_vals
            # Avoid division by zero
            denom[denom == 0] = 1.0

            img_3ch = (img_3ch - min_vals) / denom
            img_3ch = np.clip(img_3ch, 0.0, 1.0)

        # 3. Convert to Tensor and Permute to (C, H, W)
        img_tensor = torch.from_numpy(img_3ch).float().permute(2, 0, 1)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # 4. Augmentation (Training Only)
        if self.transform:
            # Random Rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            img_tensor = torch.rot90(img_tensor, k, dims=[1, 2])

            # Random Horizontal Flip
            if np.random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])

            # Vertical Flip is excluded per instructions

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            return img_tensor, angle_tensor


def make_dataloaders(train_data, fold_idx=0, batch_size=Config.BATCH_SIZE):
    """
    Creates training and validation dataloaders for a specific Stratified K-Fold.
    Computes scaling statistics based ONLY on the training fold to prevent leakage.

    Returns:
        train_loader, val_loader, scaler_stats
    """
    images = train_data["images"]
    labels = train_data["labels"]
    angles = train_data["angles"]

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the specific fold
    splits = list(skf.split(images, labels))
    if fold_idx >= len(splits):
        raise ValueError(
            f"Fold index {fold_idx} out of range for {Config.NUM_FOLDS} folds."
        )

    train_idx, val_idx = splits[fold_idx]

    # Split Data
    X_train, X_val = images[train_idx], images[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    a_train, a_val = angles[train_idx], angles[val_idx]

    # --- Compute Scaling Statistics on Training Fold ---
    # Construct 3rd channel for stats computation
    X_train_avg = (X_train[..., 0] + X_train[..., 1]) / 2.0
    X_train_avg = np.expand_dims(X_train_avg, axis=-1)
    X_train_3ch = np.concatenate([X_train, X_train_avg], axis=-1)  # (N, 75, 75, 3)

    # Compute Min/Max per channel over the training set
    flat = X_train_3ch.reshape(-1, 3)
    min_vals = np.min(flat, axis=0)
    max_vals = np.max(flat, axis=0)
    scaler_stats = (min_vals, max_vals)

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, a_train, y_train, transform=True, scaler_stats=scaler_stats
    )

    val_dataset = IcebergDataset(
        X_val, a_val, y_val, transform=False, scaler_stats=scaler_stats
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, scaler_stats


def make_test_dataloader(test_data, scaler_stats, batch_size=Config.BATCH_SIZE):
    """
    Creates a dataloader for the test set.

    Args:
        test_data (dict): Test data dictionary.
        scaler_stats (tuple): (min_vals, max_vals) to use for scaling.
                              Should match the stats used for the trained model.
    """
    test_dataset = IcebergDataset(
        test_data["images"],
        test_data["angles"],
        labels=None,
        transform=False,
        scaler_stats=scaler_stats,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
