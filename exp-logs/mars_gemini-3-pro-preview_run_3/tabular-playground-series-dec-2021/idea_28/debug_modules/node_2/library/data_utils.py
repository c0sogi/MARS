import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Cover Type prediction task.
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


def feature_engineering(df):
    """
    Applies physics-informed feature engineering.
    """
    # Avoid modifying the original dataframe
    df = df.copy()

    # 1. Cyclical Augmentation (Aspect)
    # Convert degrees to radians
    aspect_rad = df["Aspect"] * np.pi / 180.0
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)
    # Note: Raw 'Aspect' is retained as per requirements

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # sqrt(H^2 + V^2)
    df["Hydrology_Distance"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    # Elevation - Vertical_Dist preserves the absolute elevation of the water source
    df["Hydrology_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context (Mean Distance to Amenities)
    # Average of distances to Hydrology, Roadways, Fire Points
    df["Mean_Amenities_Dist"] = df[
        [
            "Horizontal_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
    ].mean(axis=1)

    return df


def process_data(load_cached_data=True):
    """
    Loads data, performs feature engineering, scaling, and caching.
    Returns processed numpy arrays.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in files.values())

    if load_cached_data and cache_exists:
        print("Loading cached data from", cache_dir)
        X_train = np.load(files["X_train"])
        y_train = np.load(files["y_train"])
        X_val = np.load(files["X_val"])
        y_val = np.load(files["y_val"])
        X_test = np.load(files["X_test"])
        test_ids = np.load(files["test_ids"])
        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing data from scratch...")

    # Load metadata parquets
    df_train = pd.read_parquet(Config.TRAIN_PATH)
    df_val = pd.read_parquet(Config.VAL_PATH)
    df_test = pd.read_parquet(Config.TEST_PATH)

    # Debugging: Limit training samples if configured
    if Config.MAX_TRAIN_SAMPLES is not None:
        df_train = df_train.iloc[: Config.MAX_TRAIN_SAMPLES]

    # Extract IDs and Targets
    y_train = df_train[Config.TARGET_COL].values
    y_val = df_val[Config.TARGET_COL].values
    test_ids = df_test[Config.ID_COL].values

    # Adjust targets to be 0-indexed (Class 1-7 -> 0-6)
    y_train = y_train - 1
    y_val = y_val - 1

    # Apply Feature Engineering
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # Define Continuous Columns (Original + Engineered)
    # We exclude ID, Target, and known Binary columns
    exclude_cols = set(Config.BINARY_COLS + [Config.ID_COL, Config.TARGET_COL])
    continuous_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Verify we have the expected binary columns
    binary_cols = [c for c in Config.BINARY_COLS if c in df_train.columns]

    # Standardization
    # Fit scaler ONLY on training data continuous columns
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(
        df_train[continuous_cols].values.astype(np.float32)
    )
    X_val_cont = scaler.transform(df_val[continuous_cols].values.astype(np.float32))
    X_test_cont = scaler.transform(df_test[continuous_cols].values.astype(np.float32))

    # Get Binary Features (No scaling)
    X_train_bin = df_train[binary_cols].values.astype(np.float32)
    X_val_bin = df_val[binary_cols].values.astype(np.float32)
    X_test_bin = df_test[binary_cols].values.astype(np.float32)

    # Concatenate
    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # Save to cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(files["X_train"], X_train)
    np.save(files["y_train"], y_train)
    np.save(files["X_val"], X_val)
    np.save(files["y_val"], y_val)
    np.save(files["X_test"], X_test)
    np.save(files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


def get_dataloaders(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    Also returns the input dimension size.
    """
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(load_cached_data)

    # Create Datasets
    train_dataset = CoverTypeDataset(X_train, y_train)
    val_dataset = CoverTypeDataset(X_val, y_val)
    test_dataset = CoverTypeDataset(X_test)  # No targets for test

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    input_dim = X_train.shape[1]

    return train_loader, val_loader, test_loader, input_dim, test_ids
