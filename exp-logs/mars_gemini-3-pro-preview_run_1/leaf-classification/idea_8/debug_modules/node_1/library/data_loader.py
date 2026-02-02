import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
)


def load_data(load_cached_data=True):
    """
    Loads the dataset, separating features, targets, and IDs.
    Implements caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from local cache first.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
            - X_train, X_val, X_test: Numpy arrays of shape (n_samples, 192)
            - y_train, y_val: Numpy arrays of shape (n_samples,) containing string labels
            - test_ids: Numpy array of shape (n_samples,) containing image IDs
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val.npy"),
        "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    }

    # Check if we should and can load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            X_train = np.load(cache_files["X_train"], allow_pickle=True)
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            X_val = np.load(cache_files["X_val"], allow_pickle=True)
            y_val = np.load(cache_files["y_val"], allow_pickle=True)
            X_test = np.load(cache_files["X_test"], allow_pickle=True)
            test_ids = np.load(cache_files["test_ids"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Loading data from metadata CSVs...")

    # Load DataFrames
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # Extract Features
    # Ensure we strictly follow the order in FEATURE_COLUMNS
    X_train = df_train[FEATURE_COLUMNS].values.astype(np.float64)
    X_val = df_val[FEATURE_COLUMNS].values.astype(np.float64)
    X_test = df_test[FEATURE_COLUMNS].values.astype(np.float64)

    # Extract Targets
    y_train = df_train[TARGET_COLUMN].values.astype(str)
    y_val = df_val[TARGET_COLUMN].values.astype(str)

    # Extract IDs for Test set
    test_ids = df_test[ID_COLUMN].values

    # Save to cache
    print(f"Saving data to cache at {WORKING_DIR}...")
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids
