import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.utils import seed_everything, worker_init_fn

# Configuration constants
CACHE_DIR = "./working/idea_5"
CACHE_FILE = "processed_data.npz"


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, transform=False):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            transform (bool): Whether to apply geometric augmentations.
        """
        self.X = torch.from_numpy(X).float()
        self.angles = torch.from_numpy(angles).float()
        self.y = torch.from_numpy(y).float() if y is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        angle = self.angles[idx]

        if self.transform:
            # Rotational Invariance: 0, 90, 180, 270 degrees
            k = np.random.randint(0, 4)
            img = torch.rot90(img, k, dims=[1, 2])

            # Random Horizontal Flip
            if np.random.random() > 0.5:
                img = torch.flip(img, dims=[2])

        if self.y is not None:
            return img, angle, self.y[idx]
        else:
            return img, angle


def _load_raw_data():
    """
    Reads raw JSON and CSV files, processes bands into images, handles imputation and scaling.
    """
    print("Processing raw data from scratch...")

    # Load metadata
    train_meta = pd.read_csv("./metadata/train.csv")
    val_meta = pd.read_csv("./metadata/val.csv")
    test_meta = pd.read_csv("./metadata/test.csv")

    # Load raw JSONs
    # Note: Loading entire JSONs into memory requires sufficient RAM (available in this env)
    with open("./input/train.json", "r") as f:
        train_json = json.load(f)
    with open("./input/test.json", "r") as f:
        test_json = json.load(f)

    # Convert to DataFrames for ID-based lookup
    df_train_raw = pd.DataFrame(train_json).set_index("id")
    df_test_raw = pd.DataFrame(test_json).set_index("id")

    def process_split(meta_df, raw_df, is_test=False):
        ids = meta_df["id"].values
        subset = raw_df.loc[ids]

        # Reshape flattened bands to (N, 75, 75)
        b1 = np.stack([np.array(b).reshape(75, 75) for b in subset["band_1"].values])
        b2 = np.stack([np.array(b).reshape(75, 75) for b in subset["band_2"].values])

        # Channel 3: Arithmetic Mean
        b3 = (b1 + b2) / 2.0

        # Stack to (N, 3, 75, 75)
        X = np.stack([b1, b2, b3], axis=1)

        # Extract angles (metadata has them parsed, but we ensure alignment)
        angles = meta_df["inc_angle"].values

        if not is_test:
            y = meta_df["is_iceberg"].values
            return X, angles, y
        else:
            return X, angles, ids

    # Process splits
    X_train, ang_train, y_train = process_split(train_meta, df_train_raw)
    X_val, ang_val, y_val = process_split(val_meta, df_train_raw)
    X_test, ang_test, ids_test = process_split(test_meta, df_test_raw, is_test=True)

    # 1. Impute Missing Incidence Angles
    # Use median from training set only to prevent leakage
    angle_median = np.nanmedian(ang_train)

    ang_train = np.where(np.isnan(ang_train), angle_median, ang_train)
    ang_val = np.where(np.isnan(ang_val), angle_median, ang_val)
    ang_test = np.where(np.isnan(ang_test), angle_median, ang_test)

    # 2. Min-Max Scaling to [0, 1]
    # Compute stats from training set only, per channel
    # X shape: (N, C, H, W)
    min_vals = X_train.min(axis=(0, 2, 3), keepdims=True)
    max_vals = X_train.max(axis=(0, 2, 3), keepdims=True)

    # Avoid division by zero
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0

    def scale(arr):
        return (arr - min_vals) / denom

    X_train = scale(X_train)
    X_val = scale(X_val)
    X_test = scale(X_test)

    return {
        "X_train": X_train,
        "ang_train": ang_train,
        "y_train": y_train,
        "X_val": X_val,
        "ang_val": ang_val,
        "y_val": y_val,
        "X_test": X_test,
        "ang_test": ang_test,
        "ids_test": ids_test,
    }


def get_data(load_cached_data=True):
    """
    Loads data from cache or processes it from raw files.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, CACHE_FILE)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["X_train"],
                data["ang_train"],
                data["y_train"],
                data["X_val"],
                data["ang_val"],
                data["y_val"],
                data["X_test"],
                data["ang_test"],
                data["ids_test"],
            )
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing...")

    # Process data
    data_dict = _load_raw_data()

    # Save to cache
    print(f"Saving processed data to {cache_path}...")
    np.savez(cache_path, **data_dict)

    return (
        data_dict["X_train"],
        data_dict["ang_train"],
        data_dict["y_train"],
        data_dict["X_val"],
        data_dict["ang_val"],
        data_dict["y_val"],
        data_dict["X_test"],
        data_dict["ang_test"],
        data_dict["ids_test"],
    )


def get_dataloaders(batch_size=32, load_cached_data=True):
    """
    Returns DataLoaders for train, validation, and test sets.
    """
    X_train, ang_train, y_train, X_val, ang_val, y_val, X_test, ang_test, ids_test = (
        get_data(load_cached_data)
    )

    # Create Datasets
    # Enable transform only for training
    train_ds = IcebergDataset(X_train, ang_train, y_train, transform=True)
    val_ds = IcebergDataset(X_val, ang_val, y_val, transform=False)
    test_ds = IcebergDataset(X_test, ang_test, transform=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        drop_last=True,  # Helps with BatchNorm stability
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, ids_test
