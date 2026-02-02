import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    FEATURES,
    TARGET_COL,
    ID_COL,
    FLOAT_PRECISION,
)


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, separating features, targets, and IDs.
    Implements caching using Parquet for DataFrames and NPY for Numpy arrays.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.
                                 If False or if files are missing, re-processes raw data.

    Returns:
        tuple: (train_data, val_data, test_data)
            train_data: (X_train, y_train, ids_train)
            val_data:   (X_val, y_val, ids_val)
            test_data:  (X_test, ids_test)
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.parquet"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "ids_train": os.path.join(CACHE_DIR, "ids_train.npy"),
        "X_val": os.path.join(CACHE_DIR, "X_val.parquet"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "ids_val": os.path.join(CACHE_DIR, "ids_val.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.parquet"),
        "ids_test": os.path.join(CACHE_DIR, "ids_test.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached dataset from {CACHE_DIR}...")
        try:
            X_train = pd.read_parquet(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            ids_train = np.load(cache_files["ids_train"], allow_pickle=True)

            X_val = pd.read_parquet(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"], allow_pickle=True)
            ids_val = np.load(cache_files["ids_val"], allow_pickle=True)

            X_test = pd.read_parquet(cache_files["X_test"])
            ids_test = np.load(cache_files["ids_test"], allow_pickle=True)

            return (
                (X_train, y_train, ids_train),
                (X_val, y_val, ids_val),
                (X_test, ids_test),
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing data...")

    # Process data from scratch
    print("Processing data from metadata CSVs...")

    # Load CSVs
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Process Training Data
    # Select specific features and cast to float32
    X_train = df_train[FEATURES].astype(FLOAT_PRECISION)
    y_train = df_train[TARGET_COL].values
    ids_train = df_train[ID_COL].values

    # Process Validation Data
    X_val = df_val[FEATURES].astype(FLOAT_PRECISION)
    y_val = df_val[TARGET_COL].values
    ids_val = df_val[ID_COL].values

    # Process Test Data
    X_test = df_test[FEATURES].astype(FLOAT_PRECISION)
    ids_test = df_test[ID_COL].values

    # Save to Cache
    print(f"Saving processed data to cache at {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    X_train.to_parquet(cache_files["X_train"], index=False)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["ids_train"], ids_train)

    X_val.to_parquet(cache_files["X_val"], index=False)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["ids_val"], ids_val)

    X_test.to_parquet(cache_files["X_test"], index=False)
    np.save(cache_files["ids_test"], ids_test)

    return (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, ids_test)
