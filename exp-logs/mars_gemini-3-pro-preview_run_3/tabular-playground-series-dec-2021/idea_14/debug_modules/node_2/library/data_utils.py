import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class FeatureEngineer:
    """
    Handles feature engineering and preprocessing for the Cover Type dataset.
    Implements Augmented Physics-Informed Engineering strategies.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        # Original continuous columns
        self.continuous_cols = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        # New engineered features to be treated as continuous
        self.new_features = [
            "Aspect_Sin",
            "Aspect_Cos",
            "Euclidean_Distance_To_Hydrology",
            "Absolute_Hydrology_Elevation",
            "Mean_Amenities_Dist",
        ]
        self.all_continuous = self.continuous_cols + self.new_features

    def engineer_features(self, df):
        """
        Applies physics-informed feature engineering.
        """
        # Create a copy to avoid SettingWithCopy warnings
        df = df.copy()

        # 1. Cyclical Augmentation for Aspect
        # We retain the raw 'Aspect' as per strategy, but add Sin/Cos representations
        df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
        df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

        # 2. Geometric Magnitude: Euclidean Distance to Hydrology
        # sqrt(Horizontal^2 + Vertical^2)
        df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

        # 3. Directional Preservation: Absolute Hydrology Elevation
        # Elevation - Vertical_Distance gives the absolute elevation of the water source
        df["Absolute_Hydrology_Elevation"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
        )

        # 4. Global Context: Mean Distance to Amenities
        df["Mean_Amenities_Dist"] = (
            df["Horizontal_Distance_To_Hydrology"]
            + df["Horizontal_Distance_To_Roadways"]
            + df["Horizontal_Distance_To_Fire_Points"]
        ) / 3.0

        return df

    def fit(self, df):
        """
        Fits the StandardScaler on the continuous features of the training set.
        """
        self.scaler.fit(df[self.all_continuous])

    def transform(self, df):
        """
        Applies standardization to continuous features.
        Binary features are left untouched.
        """
        df = df.copy()
        df[self.all_continuous] = self.scaler.transform(df[self.all_continuous])
        return df


def get_processed_data(load_cached_data=True):
    """
    Loads data, performs feature engineering, and handles caching.
    Returns a dictionary of numpy arrays.
    """
    cache_dir = Config.CACHE_DIR

    # Define cache file paths
    files = {
        "train_X": os.path.join(cache_dir, "train_X.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_X": os.path.join(cache_dir, "val_X.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_X": os.path.join(cache_dir, "test_X.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in files.values())
        if all_exist:
            print(f"Loading cached data from {cache_dir}...")
            try:
                data = {k: np.load(v) for k, v in files.items()}
                return data
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        else:
            print("Cache incomplete or missing. Recomputing...")
    else:
        print("Force reload requested. Recomputing...")

    # 2. Load raw data from metadata parquet files
    print("Loading raw data from Parquet...")
    df_train = pd.read_parquet(Config.TRAIN_PATH)
    df_val = pd.read_parquet(Config.VAL_PATH)
    df_test = pd.read_parquet(Config.TEST_PATH)

    # 3. Apply Feature Engineering
    print("Applying Feature Engineering...")
    fe = FeatureEngineer()

    df_train = fe.engineer_features(df_train)
    df_val = fe.engineer_features(df_val)
    df_test = fe.engineer_features(df_test)

    # 4. Fit and Transform Scaler
    print("Standardizing features...")
    fe.fit(df_train)

    df_train = fe.transform(df_train)
    df_val = fe.transform(df_val)
    df_test = fe.transform(df_test)

    # 5. Convert to NumPy and Format
    # Identify all feature columns (exclude ID and Target)
    feature_cols = [
        c for c in df_train.columns if c not in [Config.ID_COL, Config.TARGET_COL]
    ]

    print(f"Processing {len(feature_cols)} features...")

    # Train
    train_X = df_train[feature_cols].values.astype(np.float32)
    # Shift labels from 1-7 to 0-6
    train_y = (df_train[Config.TARGET_COL].values - 1).astype(np.int64)

    # Val
    val_X = df_val[feature_cols].values.astype(np.float32)
    val_y = (df_val[Config.TARGET_COL].values - 1).astype(np.int64)

    # Test
    test_X = df_test[feature_cols].values.astype(np.float32)
    test_ids = df_test[Config.ID_COL].values.astype(np.int64)

    # 6. Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(files["train_X"], train_X)
    np.save(files["train_y"], train_y)
    np.save(files["val_X"], val_X)
    np.save(files["val_y"], val_y)
    np.save(files["test_X"], test_X)
    np.save(files["test_ids"], test_ids)

    return {
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "test_X": test_X,
        "test_ids": test_ids,
    }


class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Cover Type task.
    """

    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Orchestrates data loading and creation of PyTorch DataLoaders.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, subsamples data for rapid testing.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    Config.set_seed(Config.SEED)

    # Get processed numpy arrays
    data = get_processed_data(load_cached_data=load_cached_data)

    train_X, train_y = data["train_X"], data["train_y"]
    val_X, val_y = data["val_X"], data["val_y"]
    test_X, test_ids = data["test_X"], data["test_ids"]

    # Handle Debug Mode
    if debug:
        limit = Config.DEBUG_SAMPLE_SIZE
        print(f"DEBUG MODE: Subsampling data to {limit} rows.")
        train_X = train_X[:limit]
        train_y = train_y[:limit]
        val_X = val_X[:limit]
        val_y = val_y[:limit]
        test_X = test_X[:limit]
        test_ids = test_ids[:limit]

    # Create Datasets
    train_ds = CoverTypeDataset(train_X, train_y)
    val_ds = CoverTypeDataset(val_X, val_y)
    test_ds = CoverTypeDataset(test_X, y=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
