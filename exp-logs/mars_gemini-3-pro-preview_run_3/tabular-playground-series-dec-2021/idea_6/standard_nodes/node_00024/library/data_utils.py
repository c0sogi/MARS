import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


def engineer_features(df):
    """
    Applies physics-informed geometric feature engineering to the dataframe.
    Calculates Euclidean Distance to Hydrology and Absolute Hydrology Elevation.
    """
    # Euclidean Distance to Hydrology = sqrt(Horizontal^2 + Vertical^2)
    # This captures the straight-line distance, which is physically more relevant than separate components.
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # Absolute Hydrology Elevation = Elevation - Vertical_Distance_To_Hydrology
    # Since Vertical_Dist = Elevation - Hydro_Elev, this recovers the absolute elevation of the water source.
    # This provides a fixed reference point independent of the cell's elevation.
    df["Absolute_Hydrology_Elevation"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # Mean Distance to Amenities (Hydrology, Roadways, Fire Points)
    # Cite solution_lesson_node_00009: Explicitly engineer composite features.
    df["Mean_Distance_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def process_data(load_cached_data=True):
    """
    Loads raw data (parquet), performs feature engineering, scaling, and formatting.
    Implements caching using numpy files to speed up subsequent runs.

    Returns:
        train_X (np.ndarray): Processed training features (float32)
        train_y (np.ndarray): Training labels (int64, 0-indexed)
        val_X (np.ndarray): Processed validation features (float32)
        val_y (np.ndarray): Validation labels (int64, 0-indexed)
        test_X (np.ndarray): Processed test features (float32)
        test_ids (np.ndarray): Test IDs (int64)
    """
    # Ensure working directory exists
    Config.setup()

    # Define cache paths based on Config
    cache_files = [
        Config.CACHE_TRAIN_X,
        Config.CACHE_TRAIN_Y,
        Config.CACHE_VAL_X,
        Config.CACHE_VAL_Y,
        Config.CACHE_TEST_X,
        Config.CACHE_TEST_IDS,
    ]

    # Check if all cache files exist
    if load_cached_data and all(os.path.exists(f) for f in cache_files):
        print("Loading processed data from cache...")
        train_X = np.load(Config.CACHE_TRAIN_X)
        train_y = np.load(Config.CACHE_TRAIN_Y)
        val_X = np.load(Config.CACHE_VAL_X)
        val_y = np.load(Config.CACHE_VAL_Y)
        test_X = np.load(Config.CACHE_TEST_X)
        test_ids = np.load(Config.CACHE_TEST_IDS)
        return train_X, train_y, val_X, val_y, test_X, test_ids

    print("Processing data from scratch...")

    # Load raw data from metadata parquet files
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # Apply Feature Engineering
    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Define column groups
    # Continuous cols now include the original ones plus the new engineered ones
    all_continuous_cols = Config.CONTINUOUS_COLS + Config.NEW_FEATURES
    binary_cols = Config.BINARY_COLS

    # Extract and Standardize Continuous Features
    # We fit the scaler ONLY on the training set to avoid data leakage
    scaler = StandardScaler()

    train_cont = df_train[all_continuous_cols].values.astype(np.float32)
    val_cont = df_val[all_continuous_cols].values.astype(np.float32)
    test_cont = df_test[all_continuous_cols].values.astype(np.float32)

    train_cont = scaler.fit_transform(train_cont)
    val_cont = scaler.transform(val_cont)
    test_cont = scaler.transform(test_cont)

    # Extract Binary Features (No scaling, kept as raw 0/1)
    train_bin = df_train[binary_cols].values.astype(np.float32)
    val_bin = df_val[binary_cols].values.astype(np.float32)
    test_bin = df_test[binary_cols].values.astype(np.float32)

    # Concatenate to form the "Wide" input vector
    train_X = np.hstack([train_cont, train_bin])
    val_X = np.hstack([val_cont, val_bin])
    test_X = np.hstack([test_cont, test_bin])

    # Process Targets
    # Shift labels from 1-7 to 0-6 for PyTorch CrossEntropyLoss
    train_y = df_train[Config.TARGET_COL].values.astype(np.int64) - 1
    val_y = df_val[Config.TARGET_COL].values.astype(np.int64) - 1

    # Process Test IDs
    test_ids = df_test[Config.ID_COL].values.astype(np.int64)

    # Save to Cache
    print("Saving processed data to cache...")
    np.save(Config.CACHE_TRAIN_X, train_X)
    np.save(Config.CACHE_TRAIN_Y, train_y)
    np.save(Config.CACHE_VAL_X, val_X)
    np.save(Config.CACHE_VAL_Y, val_y)
    np.save(Config.CACHE_TEST_X, test_X)
    np.save(Config.CACHE_TEST_IDS, test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Constructs PyTorch DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker processes for data loading.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader (DataLoader)
        val_loader (DataLoader)
        test_loader (DataLoader)
        test_ids (np.ndarray): Array of test IDs for submission.
        input_dim (int): The dimension of the input feature vector.
    """
    train_X, train_y, val_X, val_y, test_X, test_ids = process_data(load_cached_data)

    # Create TensorDatasets
    # Train and Val have targets
    train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    val_dataset = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))
    # Test dataset only has features
    test_dataset = TensorDataset(torch.from_numpy(test_X))

    # Create DataLoaders
    # Pin memory helps with transfer to GPU
    pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    input_dim = train_X.shape[1]

    return train_loader, val_loader, test_loader, test_ids, input_dim
