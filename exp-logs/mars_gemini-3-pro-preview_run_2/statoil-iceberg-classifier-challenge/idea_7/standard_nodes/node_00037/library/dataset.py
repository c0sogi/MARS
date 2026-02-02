import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


class RandomRadarTransform:
    """
    Applies random geometric transformations suitable for Radar imagery.
    - Random 90-degree rotations (0, 90, 180, 270)
    - Random Horizontal Flips
    - No Vertical Flips (as per constraints)
    """

    def __call__(self, img):
        # img shape is (H, W, C)

        # Random rotation: k is number of 90 degree rotations
        k = np.random.randint(0, 4)
        img = np.rot90(img, k, axes=(0, 1))

        # Random horizontal flip
        if np.random.random() < 0.5:
            img = np.fliplr(img)

        return np.ascontiguousarray(img)


class IcebergDataset(Dataset):
    def __init__(self, images, stats, labels=None, transform=None):
        self.images = images
        self.stats = stats
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        stat = self.stats[idx]

        if self.transform:
            img = self.transform(img)

        # Convert to tensor and permute to (C, H, W)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        stat_tensor = torch.from_numpy(stat).float()

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, stat_tensor, label_tensor
        else:
            return img_tensor, stat_tensor


def process_data(load_cached_data=True):
    """
    Loads raw JSON data, performs feature engineering, and caches the result.
    """
    Config.make_dirs()
    cache_path = Config.PROCESSED_DATA_CACHE

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        return (
            data["X_train_img"],
            data["X_train_stats"],
            data["y_train"],
            data["X_test_img"],
            data["X_test_stats"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # Load Raw Data
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # Helper to process a list of dicts into arrays
    def process_subset(data_list, is_train=True):
        images = []
        stats = []
        ids = []
        labels = []
        inc_angles = []

        for item in data_list:
            ids.append(item["id"])

            # 1. Image Processing
            # Reshape flattened bands to 75x75
            b1 = np.array(item["band_1"]).reshape(75, 75)
            b2 = np.array(item["band_2"]).reshape(75, 75)
            # Calculate Average Band
            b_avg = (b1 + b2) / 2.0

            # Stack channels: (75, 75, 3)
            img = np.dstack((b1, b2, b_avg))
            images.append(img)

            # 2. Extract Incidence Angle
            inc = item["inc_angle"]
            if inc == "na":
                inc_angles.append(np.nan)
            else:
                inc_angles.append(float(inc))

            # 3. Statistical Features Calculation
            # Explicit stats removed to avoid redundancy (Cite solution_lesson_node_00034).
            # We only use Incidence Angle (added later).

            if is_train:
                labels.append(item["is_iceberg"])

        return (
            np.array(images, dtype=np.float32),
            np.zeros((len(images), 0), dtype=np.float32),  # Empty placeholder
            np.array(inc_angles, dtype=np.float32),
            np.array(ids),
            np.array(labels, dtype=np.float32) if is_train else None,
        )

    # Process Train and Test
    X_train_img, X_train_stats_partial, train_inc, train_ids, y_train = process_subset(
        train_data, is_train=True
    )
    X_test_img, X_test_stats_partial, test_inc, test_ids, _ = process_subset(
        test_data, is_train=False
    )

    # Impute Incidence Angle
    # Calculate mean from valid training data
    inc_mean = np.nanmean(train_inc)

    # Fill NaNs
    train_inc = np.where(np.isnan(train_inc), inc_mean, train_inc)
    test_inc = np.where(np.isnan(test_inc), inc_mean, test_inc)

    # Stats is just the incidence angle
    # Cite solution_lesson_node_00034
    X_train_stats = train_inc.reshape(-1, 1)
    X_test_stats = test_inc.reshape(-1, 1)

    # Save to cache
    np.savez(
        cache_path,
        X_train_img=X_train_img,
        X_train_stats=X_train_stats,
        y_train=y_train,
        X_test_img=X_test_img,
        X_test_stats=X_test_stats,
        test_ids=test_ids,
    )

    print(f"Data processed and saved to {cache_path}")
    return X_train_img, X_train_stats, y_train, X_test_img, X_test_stats, test_ids


def get_fold_loaders(fold_idx, load_cached_data=True):
    """
    Returns train and validation loaders for a specific fold in Stratified K-Fold.
    Performs normalization strictly based on the training split of the fold.
    """
    seed_everything(Config.SEED)

    # Load data
    X_img, X_stats, y, _, _, _ = process_data(load_cached_data)

    # Debugging: Limit samples
    if Config.MAX_SAMPLES:
        X_img = X_img[: Config.MAX_SAMPLES]
        X_stats = X_stats[: Config.MAX_SAMPLES]
        y = y[: Config.MAX_SAMPLES]

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Get indices for the requested fold
    # We iterate to find the specific fold indices
    for i, (train_idx, val_idx) in enumerate(skf.split(X_img, y)):
        if i == fold_idx:
            break
    else:
        raise ValueError(f"Fold index {fold_idx} out of range (0-{Config.NUM_FOLDS-1})")

    # Split data
    X_train_img, X_val_img = X_img[train_idx], X_img[val_idx]
    X_train_stats, X_val_stats = X_stats[train_idx], X_stats[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # --- Normalization ---
    # 1. Images: Min-Max Scaling per channel based on Train
    # X_train_img shape: (N, 75, 75, 3)
    min_vals = X_train_img.min(axis=(0, 1, 2), keepdims=True)
    max_vals = X_train_img.max(axis=(0, 1, 2), keepdims=True)

    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    X_train_img = (X_train_img - min_vals) / range_vals
    X_val_img = (X_val_img - min_vals) / range_vals

    # 2. Stats: Standard Scaling based on Train
    mean_stats = X_train_stats.mean(axis=0)
    std_stats = X_train_stats.std(axis=0)
    std_stats[std_stats == 0] = 1.0

    X_train_stats = (X_train_stats - mean_stats) / std_stats
    X_val_stats = (X_val_stats - mean_stats) / std_stats

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train_img, X_train_stats, y_train, transform=RandomRadarTransform()
    )

    val_dataset = IcebergDataset(X_val_img, X_val_stats, y_val, transform=None)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns a loader for the test set.
    Scales data using statistics from the FULL training set.
    """
    seed_everything(Config.SEED)

    # Load all data
    X_train_img, X_train_stats, _, X_test_img, X_test_stats, test_ids = process_data(
        load_cached_data
    )

    # --- Normalization (Fit on Full Train) ---
    # 1. Images
    min_vals = X_train_img.min(axis=(0, 1, 2), keepdims=True)
    max_vals = X_train_img.max(axis=(0, 1, 2), keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0

    X_test_img = (X_test_img - min_vals) / range_vals

    # 2. Stats
    mean_stats = X_train_stats.mean(axis=0)
    std_stats = X_train_stats.std(axis=0)
    std_stats[std_stats == 0] = 1.0

    X_test_stats = (X_test_stats - mean_stats) / std_stats

    # Create Dataset
    test_dataset = IcebergDataset(
        X_test_img, X_test_stats, labels=None, transform=None  # No labels for test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return test_loader, test_ids
