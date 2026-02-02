import os
import pandas as pd
import numpy as np
from library import config


def load_datasets(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from the cache directory.
                                 If False or if cache is missing, processes raw metadata and saves to cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
            - X_train (pd.DataFrame): Training features (float64).
            - y_train (np.ndarray): Training labels (strings).
            - X_val (pd.DataFrame): Validation features (float64).
            - y_val (np.ndarray): Validation labels (strings).
            - X_test (pd.DataFrame): Test features (float64).
            - test_ids (np.ndarray): Test image IDs (integers).
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_paths = {
        "X_train": os.path.join(config.CACHE_DIR, "X_train.parquet"),
        "y_train": os.path.join(config.CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(config.CACHE_DIR, "X_val.parquet"),
        "y_val": os.path.join(config.CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(config.CACHE_DIR, "X_test.parquet"),
        "test_ids": os.path.join(config.CACHE_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(path) for path in cache_paths.values())

    if load_cached_data and cache_exists:
        print("Loading datasets from cache...")
        X_train = pd.read_parquet(cache_paths["X_train"])
        y_train = np.load(cache_paths["y_train"], allow_pickle=True)
        X_val = pd.read_parquet(cache_paths["X_val"])
        y_val = np.load(cache_paths["y_val"], allow_pickle=True)
        X_test = pd.read_parquet(cache_paths["X_test"])
        test_ids = np.load(cache_paths["test_ids"], allow_pickle=True)

        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing datasets from metadata CSVs...")

    # Load metadata CSVs
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(config.VAL_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH)

    # Extract Features
    # We use the deterministic list of feature columns from config to ensure order
    # and cast to float64 as per the requirement for high-precision learning.
    X_train = df_train[config.FEATURE_COLS].astype("float64")
    X_val = df_val[config.FEATURE_COLS].astype("float64")
    X_test = df_test[config.FEATURE_COLS].astype("float64")

    # Extract Targets
    y_train = df_train[config.TARGET_COL].values
    y_val = df_val[config.TARGET_COL].values

    # Extract Test IDs
    test_ids = df_test[config.ID_COL].values

    # Save to cache
    print(f"Saving processed datasets to {config.CACHE_DIR}...")
    X_train.to_parquet(cache_paths["X_train"], index=False)
    np.save(cache_paths["y_train"], y_train)

    X_val.to_parquet(cache_paths["X_val"], index=False)
    np.save(cache_paths["y_val"], y_val)

    X_test.to_parquet(cache_paths["X_test"], index=False)
    np.save(cache_paths["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids
