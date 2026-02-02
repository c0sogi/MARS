import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from library.config import Config


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering.
    Generates new features while preserving raw signals.
    """
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Cyclical Augmentation for Aspect
    # We convert degrees to radians before applying sin/cos
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # sqrt(H^2 + V^2)
    h_dist = df["Horizontal_Distance_To_Hydrology"]
    v_dist = df["Vertical_Distance_To_Hydrology"]
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(h_dist**2 + v_dist**2)

    # 3. Directional Preservation (Hydrology Elevation)
    # Elevation - Vertical Distance
    df["Hydrology_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context (Mean Distance to Amenities)
    # Mean of distances to Hydrology, Roadways, Fire Points
    amenities = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    df["Mean_Distance_To_Amenities"] = df[amenities].mean(axis=1)

    return df


def process_data(load_cached_data=True):
    """
    Loads raw parquet files, applies feature engineering and scaling,
    and caches the processed numpy arrays.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        X_train, y_train, X_val, y_val, X_test, test_ids
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check cache existence
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print("Loading cached processed data...")
        X_train = np.load(files["X_train"])
        y_train = np.load(files["y_train"])
        X_val = np.load(files["X_val"])
        y_val = np.load(files["y_val"])
        X_test = np.load(files["X_test"])
        test_ids = np.load(files["test_ids"])
        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing data from scratch...")

    # Load Metadata
    print(f"Loading train data from {Config.TRAIN_DATA_PATH}...")
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    print(f"Loading validation data from {Config.VAL_DATA_PATH}...")
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)
    print(f"Loading test data from {Config.TEST_DATA_PATH}...")
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # Extract Test IDs
    test_ids = df_test[Config.ID_COL].values

    # Apply Feature Engineering
    print("Applying feature engineering...")
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # Define columns
    cont_cols = Config.CONTINUOUS_FEATURES
    bin_cols = Config.BINARY_FEATURES

    # Validate columns
    for c in cont_cols + bin_cols:
        if c not in df_train.columns:
            raise ValueError(f"Expected column {c} not found in dataframe.")

    # Standardization (Continuous Features Only)
    print("Standardizing continuous features...")
    scaler = StandardScaler()

    # Fit on Train, Transform on all
    X_train_cont = scaler.fit_transform(df_train[cont_cols].values.astype(np.float32))
    X_val_cont = scaler.transform(df_val[cont_cols].values.astype(np.float32))
    X_test_cont = scaler.transform(df_test[cont_cols].values.astype(np.float32))

    # Extract Binary Features (No scaling)
    X_train_bin = df_train[bin_cols].values.astype(np.float32)
    X_val_bin = df_val[bin_cols].values.astype(np.float32)
    X_test_bin = df_test[bin_cols].values.astype(np.float32)

    # Concatenate Features
    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # Process Targets (Map 1-7 to 0-6)
    y_train = (df_train[Config.TARGET_COL].values - 1).astype(np.int64)
    y_val = (df_val[Config.TARGET_COL].values - 1).astype(np.int64)

    # Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["X_val"], X_val)
    np.save(files["y_val"], y_val)
    np.save(files["X_test"], X_test)
    np.save(files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True, num_workers=4):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to use cached data.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    # Get processed data
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(
        load_cached_data=load_cached_data
    )

    # Convert to Tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test))

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

    return train_loader, val_loader, test_loader, test_ids
