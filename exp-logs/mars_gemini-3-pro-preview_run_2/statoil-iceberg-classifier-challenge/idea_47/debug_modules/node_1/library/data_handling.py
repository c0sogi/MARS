import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def get_global_stats(X, inc):
    """
    Computes global statistics from the training data for normalization.
    """
    stats = {}
    # Image Stats (Min/Max per channel)
    # X shape: (N, 3, 75, 75)
    for c in range(3):
        channel_data = X[:, c, :, :]
        stats[f"ch{c}_min"] = float(np.min(channel_data))
        stats[f"ch{c}_max"] = float(np.max(channel_data))

    # Incidence Angle Stats (Mean/Std)
    stats["inc_mean"] = float(np.mean(inc))
    stats["inc_std"] = float(np.std(inc))

    return stats


def load_data(load_cached_data=True):
    """
    Loads data from JSON files and Metadata CSVs.
    Handles caching, imputation, and channel construction.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(Config.CACHE_FILE):
        print(f"Loading cached data from {Config.CACHE_FILE}")
        try:
            data = np.load(Config.CACHE_FILE, allow_pickle=True)
            # Reconstruct dict from npz
            return {key: data[key] for key in data.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch...")

    # 2. Load Metadata
    print("Loading metadata...")
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # 3. Load Raw JSONs
    print("Loading raw JSON files...")
    with open(Config.TRAIN_JSON, "r") as f:
        train_json_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_json_data = json.load(f)

    # Map ID to Data for O(1) access
    data_map = {item["id"]: item for item in train_json_data}
    data_map.update({item["id"]: item for item in test_json_data})

    # Helper to extract and process a single sample
    def extract_sample(row):
        img_id = row["id"]
        item = data_map[img_id]

        # Bands: Flattened list -> (75, 75)
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
        # 3rd Channel: Mean of B1 and B2
        b3 = (b1 + b2) / 2.0

        # Stack: (3, 75, 75)
        img = np.stack([b1, b2, b3], axis=0)

        # Incidence Angle
        inc = item["inc_angle"]
        if inc == "na":
            inc = np.nan
        else:
            inc = float(inc)

        # Target (if exists)
        target = row["is_iceberg"] if "is_iceberg" in row else -1.0

        return img, inc, target

    # 4. Process Splits
    print("Processing splits...")

    def process_split(meta_df):
        X_list, inc_list, y_list = [], [], []
        for _, row in meta_df.iterrows():
            img, inc, y = extract_sample(row)
            X_list.append(img)
            inc_list.append(inc)
            y_list.append(y)
        return (
            np.array(X_list, dtype=np.float32),
            np.array(inc_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
        )

    X_train, inc_train, y_train = process_split(train_meta)
    X_val, inc_val, y_val = process_split(val_meta)
    X_test, inc_test, _ = process_split(test_meta)  # y is dummy here

    # 5. Impute Incidence Angle
    # Compute mean from Train ONLY
    inc_mean = np.nanmean(inc_train)

    # Fill NaNs
    inc_train = np.where(np.isnan(inc_train), inc_mean, inc_train)
    inc_val = np.where(np.isnan(inc_val), inc_mean, inc_val)
    inc_test = np.where(np.isnan(inc_test), inc_mean, inc_test)

    # 6. Compute Global Statistics from Train
    print("Computing global statistics...")
    stats = get_global_stats(X_train, inc_train)

    # 7. Save to Cache
    print(f"Saving processed data to {Config.CACHE_FILE}...")
    os.makedirs(os.path.dirname(Config.CACHE_FILE), exist_ok=True)
    np.savez(
        Config.CACHE_FILE,
        X_train=X_train,
        y_train=y_train,
        inc_train=inc_train,
        X_val=X_val,
        y_val=y_val,
        inc_val=inc_val,
        X_test=X_test,
        inc_test=inc_test,
        stats=stats,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "inc_train": inc_train,
        "X_val": X_val,
        "y_val": y_val,
        "inc_val": inc_val,
        "X_test": X_test,
        "inc_test": inc_test,
        "stats": stats,
    }


class IcebergDataset(Dataset):
    def __init__(self, X, inc, y, stats, transform=False):
        self.X = X
        self.inc = inc
        self.y = y
        self.stats = stats
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load data
        img = self.X[idx].copy()  # Shape: (3, 75, 75)
        inc_angle = self.inc[idx]
        label = self.y[idx]

        # 1. Global Normalization (Min-Max)
        # Formula: (x - min) / (max - min)
        # We allow values to go beyond [0, 1] if they exceed training bounds (No Hard Clipping)
        for c in range(3):
            c_min = self.stats[f"ch{c}_min"]
            c_max = self.stats[f"ch{c}_max"]
            denom = c_max - c_min + 1e-8
            img[c, :, :] = (img[c, :, :] - c_min) / denom

        # 2. Incidence Angle Normalization (Standardization)
        # Formula: (x - mean) / std
        inc_angle = (inc_angle - self.stats["inc_mean"]) / (
            self.stats["inc_std"] + 1e-8
        )

        # Convert to Tensor
        img_tensor = torch.from_numpy(img).float()
        inc_tensor = torch.tensor(inc_angle, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        # 3. Augmentation
        if self.transform:
            # Random Rotation: 0, 90, 180, 270 degrees
            # k is number of times to rotate by 90 degrees
            k = np.random.randint(0, 4)
            if k > 0:
                img_tensor = torch.rot90(img_tensor, k, [1, 2])

            # Horizontal Flip
            # Input is (C, H, W), so we flip dimension 2 (Width)
            if np.random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])

        return img_tensor, inc_tensor, label_tensor


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug_size=None,
):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    data = load_data(load_cached_data=load_cached_data)

    # Unpack stats (handle 0-d array from npz)
    stats = data["stats"]
    if isinstance(stats, np.ndarray):
        stats = stats.item()

    X_train, y_train, inc_train = data["X_train"], data["y_train"], data["inc_train"]
    X_val, y_val, inc_val = data["X_val"], data["y_val"], data["inc_val"]
    X_test, inc_test = data["X_test"], data["inc_test"]

    # Debugging: Slice datasets
    if debug_size is not None:
        print(f"DEBUG MODE: Truncating datasets to {debug_size} samples.")
        X_train, y_train, inc_train = (
            X_train[:debug_size],
            y_train[:debug_size],
            inc_train[:debug_size],
        )
        X_val, y_val, inc_val = (
            X_val[:debug_size],
            y_val[:debug_size],
            inc_val[:debug_size],
        )
        X_test, inc_test = X_test[:debug_size], inc_test[:debug_size]

    # Create Datasets
    train_ds = IcebergDataset(X_train, inc_train, y_train, stats, transform=True)
    val_ds = IcebergDataset(X_val, inc_val, y_val, stats, transform=False)
    # Test set has no labels, pass zeros
    test_ds = IcebergDataset(
        X_test, inc_test, np.zeros(len(X_test)), stats, transform=False
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
