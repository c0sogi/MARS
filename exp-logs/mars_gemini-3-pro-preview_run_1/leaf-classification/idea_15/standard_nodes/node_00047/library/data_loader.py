import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    FEATURE_COLS,
    TARGET_COL,
    ID_COL,
)


def load_dataset(load_cached_data=True):
    """
    Loads the dataset from metadata CSVs or cache.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data
                                 from the working directory.

    Returns:
        tuple: (X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_paths = {
        "X_train": os.path.join(WORKING_DIR, "X_train.parquet"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "ids_train": os.path.join(WORKING_DIR, "ids_train.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val.parquet"),
        "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
        "ids_val": os.path.join(WORKING_DIR, "ids_val.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.parquet"),
        "ids_test": os.path.join(WORKING_DIR, "ids_test.npy"),
    }

    # Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_paths.values())
        if all_exist:
            print("Loading dataset from cache...")
            X_train = pd.read_parquet(cache_paths["X_train"])
            y_train = np.load(cache_paths["y_train"], allow_pickle=True)
            ids_train = np.load(cache_paths["ids_train"], allow_pickle=True)

            X_val = pd.read_parquet(cache_paths["X_val"])
            y_val = np.load(cache_paths["y_val"], allow_pickle=True)
            ids_val = np.load(cache_paths["ids_val"], allow_pickle=True)

            X_test = pd.read_parquet(cache_paths["X_test"])
            ids_test = np.load(cache_paths["ids_test"], allow_pickle=True)

            return X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test
        else:
            print("Cache not found or incomplete. Loading from raw metadata...")
    else:
        print("Ignoring cache. Loading from raw metadata...")

    # Load raw metadata
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Process Training Data
    # Enforce float64 precision and deterministic column order
    X_train = df_train[FEATURE_COLS].astype(np.float64)
    y_train = df_train[TARGET_COL].values
    ids_train = df_train[ID_COL].values

    # Process Validation Data
    X_val = df_val[FEATURE_COLS].astype(np.float64)
    y_val = df_val[TARGET_COL].values
    ids_val = df_val[ID_COL].values

    # Process Test Data
    X_test = df_test[FEATURE_COLS].astype(np.float64)
    ids_test = df_test[ID_COL].values

    # Save to cache
    print("Saving processed dataset to cache...")
    X_train.to_parquet(cache_paths["X_train"])
    np.save(cache_paths["y_train"], y_train)
    np.save(cache_paths["ids_train"], ids_train)

    X_val.to_parquet(cache_paths["X_val"])
    np.save(cache_paths["y_val"], y_val)
    np.save(cache_paths["ids_val"], ids_val)

    X_test.to_parquet(cache_paths["X_test"])
    np.save(cache_paths["ids_test"], ids_test)

    return X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test
