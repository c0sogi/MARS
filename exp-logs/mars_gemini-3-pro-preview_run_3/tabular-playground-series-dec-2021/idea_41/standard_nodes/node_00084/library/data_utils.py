import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from library.config import Config


class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Forest Cover Type prediction task.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def get_physics_features(df):
    """
    Generates physics-informed features based on domain knowledge.
    """
    df = df.copy()

    # Aspect transformation (Cyclical)
    # Aspect is in degrees (0-360)
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # Hydrology distances
    # Euclidean distance combines horizontal and vertical distance
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # Absolute vertical distance (magnitude matters more than direction for some vegetation)
    df["Abs_Vertical_Distance_To_Hydrology"] = np.abs(
        df["Vertical_Distance_To_Hydrology"]
    )

    # Amenities mean distance
    # Average distance to water, roads, and fire points
    df["Mean_Distance_To_Amenities"] = (
        df["Horizontal_Distance_To_Fire_Points"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Hydrology"]
    ) / 3.0

    return df


def feature_engineering(df_train, df_val, df_test):
    """
    Applies feature engineering, scaling, and distributional augmentation.
    """
    print("Starting feature engineering...")

    # 1. Generate Physics Features
    df_train = get_physics_features(df_train)
    df_val = get_physics_features(df_val)
    df_test = get_physics_features(df_test)

    # 2. Identify Column Groups
    # Binary columns (Wilderness_Area and Soil_Type)
    binary_cols = [
        c
        for c in df_train.columns
        if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
    ]

    # Continuous columns are everything else except Id and Target
    exclude_cols = [Config.ID_COL, Config.TARGET_COL] + binary_cols
    continuous_cols = [c for c in df_train.columns if c not in exclude_cols]

    print(f"Continuous features: {len(continuous_cols)}")
    print(f"Binary features: {len(binary_cols)}")

    # 3. Extract Raw Data
    X_train_cont = df_train[continuous_cols].values.astype(np.float32)
    X_val_cont = df_val[continuous_cols].values.astype(np.float32)
    X_test_cont = df_test[continuous_cols].values.astype(np.float32)

    X_train_bin = df_train[binary_cols].values.astype(np.float32)
    X_val_bin = df_val[binary_cols].values.astype(np.float32)
    X_test_bin = df_test[binary_cols].values.astype(np.float32)

    # 4. Standardization (Z-score)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_cont)
    X_val_std = scaler.transform(X_val_cont)
    X_test_std = scaler.transform(X_test_cont)

    # 5. Distributional Augmentation (Quantile Transform -> Gaussian)
    # This creates a second view of the continuous features
    if Config.USE_QUANTILE_TRANSFORM:
        print("Applying Quantile Transformation (Gaussian)...")
        qt = QuantileTransformer(
            output_distribution=Config.QUANTILE_OUTPUT_DIST,
            n_quantiles=min(1000, len(df_train)),
            random_state=Config.SEED,
            subsample=min(100000, len(df_train)),  # efficient subsampling
        )
        X_train_qt = qt.fit_transform(X_train_cont)
        X_val_qt = qt.transform(X_val_cont)
        X_test_qt = qt.transform(X_test_cont)

        # Concatenate: [Standardized, Quantile, Binary]
        train_X = np.hstack([X_train_std, X_train_qt, X_train_bin])
        val_X = np.hstack([X_val_std, X_val_qt, X_val_bin])
        test_X = np.hstack([X_test_std, X_test_qt, X_test_bin])
    else:
        # Concatenate: [Standardized, Binary]
        train_X = np.hstack([X_train_std, X_train_bin])
        val_X = np.hstack([X_val_std, X_val_bin])
        test_X = np.hstack([X_test_std, X_test_bin])

    # 6. Process Targets
    # Map class labels (e.g., 1, 2, 3, 4, 6, 7) to 0-5
    train_y = (
        df_train[Config.TARGET_COL].map(Config.CLASS_MAPPING).values.astype(np.int64)
    )
    val_y = df_val[Config.TARGET_COL].map(Config.CLASS_MAPPING).values.astype(np.int64)

    # Extract Test IDs for submission
    test_ids = df_test[Config.ID_COL].values.astype(np.int64)

    print(f"Processed Train Shape: {train_X.shape}")
    print(f"Processed Val Shape: {val_X.shape}")
    print(f"Processed Test Shape: {test_X.shape}")

    return train_X, train_y, val_X, val_y, test_X, test_ids


def load_data(load_cached_data=True):
    """
    Loads data, handling caching logic.
    """
    # Define cache paths
    cache_files = {
        "train_X": Config.CACHE_TRAIN_X,
        "train_y": Config.CACHE_TRAIN_Y,
        "val_X": Config.CACHE_VAL_X,
        "val_y": Config.CACHE_VAL_Y,
        "test_X": Config.CACHE_TEST_X,
        "test_ids": Config.CACHE_TEST_IDS,
    }

    # Check if we can load from cache
    all_cache_exists = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and all_cache_exists:
        print("Loading data from cache...")
        train_X = np.load(Config.CACHE_TRAIN_X)
        train_y = np.load(Config.CACHE_TRAIN_Y)
        val_X = np.load(Config.CACHE_VAL_X)
        val_y = np.load(Config.CACHE_VAL_Y)
        test_X = np.load(Config.CACHE_TEST_X)
        test_ids = np.load(Config.CACHE_TEST_IDS)
        return train_X, train_y, val_X, val_y, test_X, test_ids

    # If not cached or forced reload, process from scratch
    print("Loading raw data from parquet metadata...")
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # Debug mode: subsample data
    if Config.DEBUG:
        print("DEBUG MODE: Subsampling data...")
        df_train = df_train.iloc[:10000]
        df_val = df_val.iloc[:2000]
        df_test = df_test.iloc[:2000]

    # Run feature engineering
    train_X, train_y, val_X, val_y, test_X, test_ids = feature_engineering(
        df_train, df_val, df_test
    )

    # Save to cache
    print("Saving processed data to cache...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(Config.CACHE_TRAIN_X, train_X)
    np.save(Config.CACHE_TRAIN_Y, train_y)
    np.save(Config.CACHE_VAL_X, val_X)
    np.save(Config.CACHE_VAL_Y, val_y)
    np.save(Config.CACHE_TEST_X, test_X)
    np.save(Config.CACHE_TEST_IDS, test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids
