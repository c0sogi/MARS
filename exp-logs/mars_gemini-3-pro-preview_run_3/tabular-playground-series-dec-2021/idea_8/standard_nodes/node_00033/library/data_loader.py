import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import library.config as config


# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------
def engineer_features(df):
    """
    Applies the Cyclical-Geometric Feature Paradigm.
    Transforms raw features into the final feature set defined in config.
    """
    df_eng = df.copy()

    # 1. Cyclical Topology: Aspect
    # Convert degrees to radians
    aspect_rad = np.radians(df_eng["Aspect"])
    df_eng["Aspect_Sin"] = np.sin(aspect_rad)
    df_eng["Aspect_Cos"] = np.cos(aspect_rad)

    # 2. Geometric Magnitude: Euclidean Distance to Hydrology
    # sqrt(h_dist^2 + v_dist^2)
    h_dist_hydro = df_eng["Horizontal_Distance_To_Hydrology"]
    v_dist_hydro = df_eng["Vertical_Distance_To_Hydrology"]
    df_eng["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        h_dist_hydro**2 + v_dist_hydro**2
    )

    # 3. Directional Preservation: Absolute Hydrology Elevation
    # Elevation - Vertical_Distance_To_Hydrology
    df_eng["Absolute_Hydrology_Elevation"] = df_eng["Elevation"] - v_dist_hydro

    # 4. Global Remoteness: Mean Distance to Amenities
    # Mean of (Hydrology, Roadways, Fire Points)
    # Note: Using horizontal distances
    dists = [
        df_eng["Horizontal_Distance_To_Hydrology"],
        df_eng["Horizontal_Distance_To_Roadways"],
        df_eng["Horizontal_Distance_To_Fire_Points"],
    ]
    df_eng["Mean_Distance_To_Amenities"] = np.mean(dists, axis=0)

    # Ensure we only return the columns specified in config
    # Concatenate Continuous and Binary features
    # Note: Binary features are already in the dataframe, we just select them.

    # Check if all required columns exist
    required_cols = config.FINAL_CONTINUOUS_FEATURES + config.FINAL_BINARY_FEATURES

    # Return selected columns.
    # If target column exists in input, we don't include it in the returned features df,
    # it is handled separately in the processing pipeline.
    return df_eng[required_cols]


# -----------------------------------------------------------------------------
# Data Processing & Caching
# -----------------------------------------------------------------------------
def get_data_arrays(load_cached_data=True):
    """
    Loads data from cache or computes it from scratch.
    Returns numpy arrays: X_train, y_train, X_val, y_val, X_test
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_X": os.path.join(cache_dir, "train_X.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_X": os.path.join(cache_dir, "val_X.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_X": os.path.join(cache_dir, "test_X.npy"),
    }

    # Try loading from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in files.values())
        if all_exist:
            print("Loading data from cache...")
            return (
                np.load(files["train_X"]),
                np.load(files["train_y"]),
                np.load(files["val_X"]),
                np.load(files["val_y"]),
                np.load(files["test_X"]),
            )
        else:
            print("Cache missing or incomplete. Processing from scratch...")

    # Process from scratch
    print("Loading Parquet metadata...")
    df_train = pd.read_parquet(config.TRAIN_PATH)
    df_val = pd.read_parquet(config.VAL_PATH)
    df_test = pd.read_parquet(config.TEST_PATH)

    # Extract Targets (Shift 1-7 to 0-6 for PyTorch)
    y_train = (df_train[config.TARGET_COL] - 1).values.astype(np.int64)
    y_val = (df_val[config.TARGET_COL] - 1).values.astype(np.int64)

    # Feature Engineering
    print("Engineering features...")
    df_train_eng = engineer_features(df_train)
    df_val_eng = engineer_features(df_val)
    df_test_eng = engineer_features(df_test)

    # Scaling Continuous Features
    # We need to separate continuous and binary to scale only continuous
    print("Scaling continuous features...")
    cont_cols = config.FINAL_CONTINUOUS_FEATURES
    bin_cols = config.FINAL_BINARY_FEATURES

    scaler = StandardScaler()

    # Fit on Train
    train_cont = scaler.fit_transform(df_train_eng[cont_cols].values.astype(np.float32))
    val_cont = scaler.transform(df_val_eng[cont_cols].values.astype(np.float32))
    test_cont = scaler.transform(df_test_eng[cont_cols].values.astype(np.float32))

    # Binary features (already 0/1, just cast to float32 for consistency in concatenation)
    train_bin = df_train_eng[bin_cols].values.astype(np.float32)
    val_bin = df_val_eng[bin_cols].values.astype(np.float32)
    test_bin = df_test_eng[bin_cols].values.astype(np.float32)

    # Concatenate
    X_train = np.hstack([train_cont, train_bin])
    X_val = np.hstack([val_cont, val_bin])
    X_test = np.hstack([test_cont, test_bin])

    # Save to cache
    print("Saving processed data to cache...")
    np.save(files["train_X"], X_train)
    np.save(files["train_y"], y_train)
    np.save(files["val_X"], X_val)
    np.save(files["val_y"], y_val)
    np.save(files["test_X"], X_test)

    return X_train, y_train, X_val, y_val, X_test


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class CoverTypeDataset(Dataset):
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
        else:
            return self.X[idx]


# -----------------------------------------------------------------------------
# DataLoader Factory
# -----------------------------------------------------------------------------
def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Main entry point to get DataLoaders.
    """
    # Use config defaults if not provided
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    # Get processed numpy arrays
    X_train, y_train, X_val, y_val, X_test = get_data_arrays(
        load_cached_data=load_cached_data
    )

    # Create Datasets
    train_dataset = CoverTypeDataset(X_train, y_train)
    val_dataset = CoverTypeDataset(X_val, y_val)
    test_dataset = CoverTypeDataset(X_test, y=None)

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
