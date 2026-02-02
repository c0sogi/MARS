import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ForestDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering.
    """
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Cyclical Augmentation (Retain raw Aspect as well)
    # Aspect is in degrees (0-360)
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude
    # Euclidean Distance to Hydrology
    h_dist = df["Horizontal_Distance_To_Hydrology"]
    v_dist = df["Vertical_Distance_To_Hydrology"]
    df["Hydro_Euclidean"] = np.sqrt(h_dist**2 + v_dist**2)

    # 3. Directional Preservation
    # Absolute Hydrology Elevation: Elevation - Vertical_Distance_To_Hydrology
    # This preserves the physical elevation of the water source relative to sea level
    df["Hydro_Elevation_Diff"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context
    # Mean Distance to Amenities (Hydrology, Roadways, Fire Points)
    amenities = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    df["Amenities_Mean_Dist"] = df[amenities].mean(axis=1)

    return df


def get_test_ids():
    """
    Helper to retrieve test IDs, preferably from cache.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "test_ids.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    else:
        # Fallback if not cached
        df_test = pd.read_parquet(Config.TEST_PATH)
        return df_test["Id"].values


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    max_samples=None,
):
    """
    Loads data, performs feature engineering/preprocessing, and returns DataLoaders.
    Implements caching mechanism to save processed numpy arrays.

    Args:
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoaders.
        load_cached_data (bool): Whether to attempt loading from cache.
        max_samples (int, optional): If set, truncates dataset for debugging/fast prototyping.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_X": os.path.join(Config.CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(Config.CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(Config.CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.CACHE_DIR}...")
        X_train = np.load(cache_files["train_X"])
        y_train = np.load(cache_files["train_y"])
        X_val = np.load(cache_files["val_X"])
        y_val = np.load(cache_files["val_y"])
        X_test = np.load(cache_files["test_X"])
    else:
        print("Processing data from scratch...")

        # Load raw data from metadata
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        val_df = pd.read_parquet(Config.VAL_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)

        # Save Test IDs for submission
        test_ids = test_df["Id"].values
        np.save(cache_files["test_ids"], test_ids)

        # Extract targets (0-indexed)
        # Classes are 1-7, map to 0-6
        y_train = train_df["Cover_Type"].values - 1
        y_val = val_df["Cover_Type"].values - 1

        # Drop non-feature columns
        drop_cols_train = ["Id", "Cover_Type"]
        drop_cols_test = ["Id"]

        X_train_df = train_df.drop(columns=drop_cols_train, errors="ignore")
        X_val_df = val_df.drop(columns=drop_cols_train, errors="ignore")
        X_test_df = test_df.drop(columns=drop_cols_test, errors="ignore")

        # Apply Feature Engineering
        X_train_df = feature_engineering(X_train_df)
        X_val_df = feature_engineering(X_val_df)
        X_test_df = feature_engineering(X_test_df)

        # Identify Continuous vs Binary features
        # Binary features are Soil_Type* and Wilderness_Area*
        all_cols = X_train_df.columns
        binary_cols = [
            c
            for c in all_cols
            if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
        ]
        continuous_cols = [c for c in all_cols if c not in binary_cols]

        # Standardization (Fit on Train, Transform All)
        scaler = StandardScaler()

        # Extract continuous parts
        X_train_cont = X_train_df[continuous_cols].values.astype(np.float32)
        X_val_cont = X_val_df[continuous_cols].values.astype(np.float32)
        X_test_cont = X_test_df[continuous_cols].values.astype(np.float32)

        # Fit and Transform
        X_train_cont = scaler.fit_transform(X_train_cont)
        X_val_cont = scaler.transform(X_val_cont)
        X_test_cont = scaler.transform(X_test_cont)

        # Extract binary parts (keep as is)
        X_train_bin = X_train_df[binary_cols].values.astype(np.float32)
        X_val_bin = X_val_df[binary_cols].values.astype(np.float32)
        X_test_bin = X_test_df[binary_cols].values.astype(np.float32)

        # Concatenate
        X_train = np.hstack([X_train_cont, X_train_bin])
        X_val = np.hstack([X_val_cont, X_val_bin])
        X_test = np.hstack([X_test_cont, X_test_bin])

        # Cache the processed arrays
        np.save(cache_files["train_X"], X_train)
        np.save(cache_files["train_y"], y_train)
        np.save(cache_files["val_X"], X_val)
        np.save(cache_files["val_y"], y_val)
        np.save(cache_files["test_X"], X_test)

    # Debugging / Subsampling
    if max_samples is not None:
        print(f"Subsampling dataset to {max_samples} samples for debugging.")
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
        X_val = X_val[:max_samples]
        y_val = y_val[:max_samples]
        X_test = X_test[:max_samples]

    # Create Datasets
    train_dataset = ForestDataset(X_train, y_train)
    val_dataset = ForestDataset(X_val, y_val)
    test_dataset = ForestDataset(X_test, y=None)

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

    return train_loader, val_loader, test_loader
