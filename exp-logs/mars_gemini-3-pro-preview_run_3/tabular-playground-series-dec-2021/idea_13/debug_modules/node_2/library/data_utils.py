import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def engineer_features(df):
    """
    Applies Augmented Physics-Informed Engineering transformations.
    """
    # Avoid modifying the original dataframe
    df = df.copy()

    # 1. Cyclical Augmentation
    # Convert Aspect (degrees) to radians for trigonometric functions
    aspect_rad = df["Aspect"] * np.pi / 180.0
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)

    # 2. Geometric Magnitude
    # Euclidean Distance to Hydrology = sqrt(Horizontal^2 + Vertical^2)
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation
    # Absolute Hydrology Elevation = Elevation - Vertical_Distance_To_Hydrology
    # This recovers the absolute elevation of the water source
    df["Abs_Hydrology_Elevation"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Global Context
    # Mean Distance to Amenities (Hydrology, Roadways, Fire Points)
    df["Mean_Amenities_Dist"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def process_data(load_cached_data=True):
    """
    Loads data from metadata, performs feature engineering and preprocessing,
    and caches the result as numpy arrays.

    Returns:
        X_train, y_train, X_val, y_val, X_test, test_ids
    """
    # Ensure cache directory exists
    cache_dir = Config.WORKING_DIR
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
    all_cached = all(os.path.exists(p) for p in files.values())

    if load_cached_data and all_cached:
        print(f"Loading cached data from {cache_dir}...")
        X_train = np.load(files["train_X"])
        y_train = np.load(files["train_y"])
        X_val = np.load(files["val_X"])
        y_val = np.load(files["val_y"])
        X_test = np.load(files["test_X"])
        test_ids = np.load(files["test_ids"])
        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing data from scratch...")

    # Load data using metadata paths
    print(f"Loading train data from {Config.TRAIN_DATA_PATH}...")
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    print(f"Loading val data from {Config.VAL_DATA_PATH}...")
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)
    print(f"Loading test data from {Config.TEST_DATA_PATH}...")
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # Extract Test IDs
    test_ids = df_test[Config.ID_COL].values

    # Apply Feature Engineering
    print("Applying Augmented Physics-Informed Engineering...")
    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Separate Continuous and Binary Features
    cont_cols = Config.CONT_FEATURES
    bin_cols = Config.BINARY_FEATURES

    print("Extracting features...")
    # Continuous
    X_train_cont = df_train[cont_cols].values.astype(np.float32)
    X_val_cont = df_val[cont_cols].values.astype(np.float32)
    X_test_cont = df_test[cont_cols].values.astype(np.float32)

    # Binary (Raw)
    X_train_bin = df_train[bin_cols].values.astype(np.float32)
    X_val_bin = df_val[bin_cols].values.astype(np.float32)
    X_test_bin = df_test[bin_cols].values.astype(np.float32)

    # Standardize Continuous Features
    # Fit only on training data to avoid leakage
    print("Standardizing continuous features...")
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_cont)
    X_val_cont = scaler.transform(X_val_cont)
    X_test_cont = scaler.transform(X_test_cont)

    # Concatenate Scaled Continuous + Raw Binary
    print("Concatenating feature sets...")
    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # Process Targets
    # Shift labels from 1-7 to 0-6 for PyTorch CrossEntropyLoss
    print("Processing targets...")
    y_train = (df_train[Config.TARGET_COL].values - 1).astype(np.int64)
    y_val = (df_val[Config.TARGET_COL].values - 1).astype(np.int64)

    # Cache the processed data
    print(f"Caching processed data to {cache_dir}...")
    np.save(files["train_X"], X_train)
    np.save(files["train_y"], y_train)
    np.save(files["val_X"], X_val)
    np.save(files["val_y"], y_val)
    np.save(files["test_X"], X_test)
    np.save(files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates and returns PyTorch DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    X_train, y_train, X_val, y_val, X_test, _ = process_data(
        load_cached_data=load_cached_data
    )

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    test_dataset = TabularDataset(X_test)

    # Use pin_memory=True for faster transfer to GPU
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
