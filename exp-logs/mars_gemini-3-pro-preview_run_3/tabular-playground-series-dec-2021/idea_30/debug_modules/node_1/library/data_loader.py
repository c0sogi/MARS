import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    RAW_CONTINUOUS_FEATURES,
    RAW_BINARY_FEATURES,
    DERIVED_FEATURES,
    TARGET_COL,
    ID_COL,
    BATCH_SIZE,
    SEED,
    DEBUG,
    DEBUG_SUBSET_SIZE,
)
from library.utils import seed_everything


def feature_engineering(df):
    """
    Applies physics-informed feature engineering to the dataframe.
    """
    df = df.copy()

    # 1. Cyclical Augmentation (Aspect)
    # Aspect is in degrees (0-360). Convert to radians for trig functions.
    aspect_rad = np.deg2rad(df["Aspect"])
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # sqrt(Horizontal^2 + Vertical^2)
    df["Hydrology_Distance"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    # Elevation - Vertical_Distance
    df["Hydrology_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context (Mean Distance to Amenities)
    # Mean of Horizontal Distances to Hydrology, Roadways, Fire Points
    amenity_cols = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    df["Mean_Amenities"] = df[amenity_cols].mean(axis=1)

    return df


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches data.

    Logic:
    1. Check if cache exists and load_cached_data is True.
    2. If yes, return cached numpy arrays.
    3. If no, load parquet, process features, scale, and save to cache.
    """
    seed_everything(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val.npy"),
        "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading cached data from", WORKING_DIR)
        X_train = np.load(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"])
        X_val = np.load(cache_files["X_val"])
        y_val = np.load(cache_files["y_val"])
        X_test = np.load(cache_files["X_test"])
        test_ids = np.load(cache_files["test_ids"])
        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing data from scratch...")

    # Load Metadata
    df_train = pd.read_parquet(TRAIN_PATH)
    df_val = pd.read_parquet(VAL_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    # Debug Subsampling
    if DEBUG:
        print(f"DEBUG Mode: Subsampling to {DEBUG_SUBSET_SIZE} rows.")
        df_train = df_train.iloc[:DEBUG_SUBSET_SIZE]
        df_val = df_val.iloc[:DEBUG_SUBSET_SIZE]
        df_test = df_test.iloc[:DEBUG_SUBSET_SIZE]

    # Extract IDs for test set
    test_ids = df_test[ID_COL].values

    # Extract Targets (and shift to 0-indexed: 1-7 -> 0-6)
    y_train = df_train[TARGET_COL].values - 1
    y_val = df_val[TARGET_COL].values - 1

    # Feature Engineering
    print("Applying Feature Engineering...")
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # Define Feature Groups
    # Continuous features to be standardized
    cont_features = RAW_CONTINUOUS_FEATURES + DERIVED_FEATURES
    # Binary features to be kept as is
    bin_features = RAW_BINARY_FEATURES

    # Standardization
    print("Standardizing Continuous Features...")
    scaler = StandardScaler()

    # Fit on Train, Transform all
    X_train_cont = scaler.fit_transform(
        df_train[cont_features].values.astype(np.float32)
    )
    X_val_cont = scaler.transform(df_val[cont_features].values.astype(np.float32))
    X_test_cont = scaler.transform(df_test[cont_features].values.astype(np.float32))

    # Extract Binary Features
    X_train_bin = df_train[bin_features].values.astype(np.float32)
    X_val_bin = df_val[bin_features].values.astype(np.float32)
    X_test_bin = df_test[bin_features].values.astype(np.float32)

    # Concatenate
    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # Save to Cache
    print("Saving processed data to cache...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


class CoverTypeDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long() if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_dataloaders(load_cached_data=True, batch_size=BATCH_SIZE):
    """
    Returns dataloaders for train, val, and test sets.
    """
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(load_cached_data)

    train_ds = CoverTypeDataset(X_train, y_train)
    val_ds = CoverTypeDataset(X_val, y_val)
    test_ds = CoverTypeDataset(X_test, y=None)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader, test_ids
