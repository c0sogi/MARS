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
        """
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray, optional): Target labels.
        """
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
    Applies Augmented Physics-Informed Engineering to the dataframe.

    Transformations:
    1. Cyclical Aspect (Sin/Cos) - retaining raw Aspect.
    2. Euclidean Distance to Hydrology.
    3. Absolute Hydrology Elevation.
    4. Mean Distance to Amenities.
    """
    # Avoid modifying original dataframe
    df = df.copy()

    # 1. Cyclical Augmentation
    # Aspect is in degrees (0-360). Convert to radians.
    aspect_rad = df["Aspect"] * np.pi / 180.0
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)

    # 2. Geometric Magnitude: Euclidean Distance to Hydrology
    # sqrt(Horizontal^2 + Vertical^2)
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation: Absolute Hydrology Elevation
    # Elevation - Vertical_Distance_To_Hydrology
    df["Absolute_Hydrology_Elevation"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Global Context: Mean Distance to Amenities
    # Mean of distances to Hydrology, Roadways, Fire Points
    amenity_cols = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    df["Mean_Distance_To_Amenities"] = df[amenity_cols].mean(axis=1)

    return df


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches the data.

    Logic:
    1. Check if cache exists and load_cached_data is True.
    2. If not, load parquet files from metadata.
    3. Apply feature engineering.
    4. Standardize continuous features (fit on Train, transform all).
    5. Save to cache.

    Returns:
        X_train, y_train, X_val, y_val, X_test, test_ids
    """
    Config.create_directories()

    # Check if cache exists
    cache_files = [
        Config.CACHE_TRAIN_X,
        Config.CACHE_TRAIN_Y,
        Config.CACHE_VAL_X,
        Config.CACHE_VAL_Y,
        Config.CACHE_TEST_X,
        Config.CACHE_TEST_IDS,
    ]

    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        X_train = np.load(Config.CACHE_TRAIN_X)
        y_train = np.load(Config.CACHE_TRAIN_Y)
        X_val = np.load(Config.CACHE_VAL_X)
        y_val = np.load(Config.CACHE_VAL_Y)
        X_test = np.load(Config.CACHE_TEST_X)
        test_ids = np.load(Config.CACHE_TEST_IDS)
        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing data from scratch...")

    # Load raw data
    train_df = pd.read_parquet(Config.TRAIN_DATA_PATH)
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)
    test_df = pd.read_parquet(Config.TEST_DATA_PATH)

    # Extract IDs and Targets
    # Train/Val have Cover_Type, Test does not
    y_train = train_df["Cover_Type"].values
    y_val = val_df["Cover_Type"].values
    test_ids = test_df["Id"].values

    # Shift targets to 0-indexed (original 1-7 -> 0-6)
    y_train = y_train - 1
    y_val = y_val - 1

    # Drop Id and Target columns to isolate features
    # Note: Test set only has Id, no Cover_Type
    X_train_df = train_df.drop(columns=["Id", "Cover_Type"])
    X_val_df = val_df.drop(columns=["Id", "Cover_Type"])
    X_test_df = test_df.drop(columns=["Id"])

    # Apply Feature Engineering
    print("Applying Augmented Physics-Informed Engineering...")
    X_train_df = feature_engineering(X_train_df)
    X_val_df = feature_engineering(X_val_df)
    X_test_df = feature_engineering(X_test_df)

    # Identify Continuous vs Binary Columns
    # Binary columns are Soil_TypeX and Wilderness_AreaX
    # We can identify them by name pattern or values. Name pattern is safer here.
    all_cols = X_train_df.columns.tolist()
    binary_cols = [
        c
        for c in all_cols
        if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
    ]
    continuous_cols = [c for c in all_cols if c not in binary_cols]

    print(f"Continuous features: {len(continuous_cols)}")
    print(f"Binary features: {len(binary_cols)}")

    # Standardization
    # Fit scaler ONLY on training data continuous columns
    scaler = StandardScaler()
    scaler.fit(X_train_df[continuous_cols])

    # Transform all sets
    # We process continuous and binary separately then concat
    def transform_split(df):
        cont_data = scaler.transform(df[continuous_cols])
        # Binary data is already 0/1, no scaling needed, just ensure numpy array
        bin_data = df[binary_cols].values
        return np.hstack([cont_data, bin_data]).astype(np.float32)

    X_train = transform_split(X_train_df)
    X_val = transform_split(X_val_df)
    X_test = transform_split(X_test_df)

    # Cache the processed data
    print("Saving processed data to cache...")
    np.save(Config.CACHE_TRAIN_X, X_train)
    np.save(Config.CACHE_TRAIN_Y, y_train)
    np.save(Config.CACHE_VAL_X, X_val)
    np.save(Config.CACHE_VAL_Y, y_val)
    np.save(Config.CACHE_TEST_X, X_test)
    np.save(Config.CACHE_TEST_IDS, test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(load_cached_data)

    # Update INPUT_DIM in Config dynamically based on processed data
    # This is a bit of a hack since Config is a class, but it's useful for the model
    Config.INPUT_DIM = X_train.shape[1]

    train_dataset = CoverTypeDataset(X_train, y_train)
    val_dataset = CoverTypeDataset(X_val, y_val)
    test_dataset = CoverTypeDataset(X_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader, test_ids
