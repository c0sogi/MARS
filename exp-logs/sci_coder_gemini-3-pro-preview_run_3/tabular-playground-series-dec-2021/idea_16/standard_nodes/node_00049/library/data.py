import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config, DataConfig, TrainConfig


class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Cover Type data.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Targets are expected to be 0-indexed for CrossEntropyLoss
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering to the dataframe.

    Transformations:
    1. Cyclical Augmentation: Aspect_Sin, Aspect_Cos (keeping raw Aspect).
    2. Geometric Magnitude: Euclidean Distance to Hydrology.
    3. Directional Preservation: Elevation - Vertical_Distance_To_Hydrology.
    4. Global Context: Mean Distance to Amenities.
    """
    # 1. Cyclical Augmentation
    # Convert Aspect (degrees) to radians for trig functions
    aspect_rad = np.radians(df["Aspect"])
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # sqrt(Horizontal^2 + Vertical^2)
    h_dist = df["Horizontal_Distance_To_Hydrol"]
    v_dist = df["Vertical_Distance_To_Hydrolog"]
    df["Hydrology_Distance"] = np.sqrt(h_dist**2 + v_dist**2)

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    # Elevation - Vertical_Distance (preserves uphill/downhill context)
    df["Hydrology_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrolog"]

    # 4. Global Context (Mean Distance to Amenities)
    amenities = [
        "Horizontal_Distance_To_Hydrol",
        "Horizontal_Distance_To_Roadwa",
        "Horizontal_Distance_To_Fire_P",
    ]
    df["Mean_Amenities_Dist"] = df[amenities].mean(axis=1)

    return df


def get_processed_data(load_cached_data=True):
    """
    Loads raw data, applies feature engineering and preprocessing, and caches the result.
    If cache exists and load_cached_data is True, loads from disk.

    Returns:
        dict: Contains train_X, train_y, val_X, val_y, test_X, test_ids
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "train_X": os.path.join(cache_dir, "train_X.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_X": os.path.join(cache_dir, "val_X.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_X": os.path.join(cache_dir, "test_X.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        data = {k: np.load(v) for k, v in files.items()}
        return data

    print("Processing data from scratch...")

    # Load raw data using metadata paths
    train_df = pd.read_parquet(Config.TRAIN_DATA_PATH)
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)
    test_df = pd.read_parquet(Config.TEST_DATA_PATH)

    # Extract IDs for test set
    test_ids = test_df[DataConfig.ID_COL].values

    # Extract Targets and shift to 0-indexed (1-7 -> 0-6)
    train_y = train_df[DataConfig.TARGET_COL].values - 1
    val_y = val_df[DataConfig.TARGET_COL].values - 1

    # Apply Feature Engineering
    print("Applying feature engineering...")
    train_df = feature_engineering(train_df)
    val_df = feature_engineering(val_df)
    test_df = feature_engineering(test_df)

    # Preprocessing
    cont_cols = DataConfig.CONT_COLS
    bin_cols = DataConfig.BINARY_COLS

    # 1. Standardize Continuous Features
    # Fit only on training data to avoid leakage
    scaler = StandardScaler()
    train_cont = scaler.fit_transform(train_df[cont_cols].values.astype(np.float32))
    val_cont = scaler.transform(val_df[cont_cols].values.astype(np.float32))
    test_cont = scaler.transform(test_df[cont_cols].values.astype(np.float32))

    # 2. Binary Features (keep as is, but cast to float32)
    train_bin = train_df[bin_cols].values.astype(np.float32)
    val_bin = val_df[bin_cols].values.astype(np.float32)
    test_bin = test_df[bin_cols].values.astype(np.float32)

    # Concatenate features
    train_X = np.hstack([train_cont, train_bin])
    val_X = np.hstack([val_cont, val_bin])
    test_X = np.hstack([test_cont, test_bin])

    # Save to cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(files["train_X"], train_X)
    np.save(files["train_y"], train_y)
    np.save(files["val_X"], val_X)
    np.save(files["val_y"], val_y)
    np.save(files["test_X"], test_X)
    np.save(files["test_ids"], test_ids)

    data = {
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "test_X": test_X,
        "test_ids": test_ids,
    }
    return data


def get_dataloaders(
    batch_size=TrainConfig.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for training/inference.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # Get processed data (either from cache or computed)
    data = get_processed_data(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = CoverTypeDataset(data["train_X"], data["train_y"])
    val_dataset = CoverTypeDataset(data["val_X"], data["val_y"])
    test_dataset = CoverTypeDataset(data["test_X"], None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, data["test_ids"]
