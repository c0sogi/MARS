import os
import pandas as pd
import numpy as np
from library.config import TRAIN_PATH, VAL_PATH, TEST_PATH, WORKING_DIR, FEATURE_COLUMNS


def load_data(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.
    Implements caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids)
            - X_*: pandas DataFrame containing the 192 features in strict alphanumeric order.
            - y_*: numpy array containing the target labels (species).
            - *_ids: numpy array containing the image IDs.
    """

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train.parquet"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "train_ids": os.path.join(WORKING_DIR, "train_ids.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val.parquet"),
        "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
        "val_ids": os.path.join(WORKING_DIR, "val_ids.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.parquet"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    }

    # Check if we should and can load from cache
    all_cache_exists = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and all_cache_exists:
        print("Loading data from cache...")
        try:
            X_train = pd.read_parquet(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            train_ids = np.load(cache_files["train_ids"])

            X_val = pd.read_parquet(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"], allow_pickle=True)
            val_ids = np.load(cache_files["val_ids"])

            X_test = pd.read_parquet(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"])

            return X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # Process from source
    print("Loading data from metadata CSVs...")

    # Load raw CSVs
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Extract IDs
    train_ids = df_train["id"].values
    val_ids = df_val["id"].values
    test_ids = df_test["id"].values

    # Extract Targets (Species)
    y_train = df_train["species"].values
    y_val = df_val["species"].values
    # Test set does not have species column

    # Extract Features
    # Strictly filter and reorder columns based on config.FEATURE_COLUMNS
    # This ensures deterministic column order for the solver
    X_train = df_train[FEATURE_COLUMNS].copy()
    X_val = df_val[FEATURE_COLUMNS].copy()
    X_test = df_test[FEATURE_COLUMNS].copy()

    # Save to cache
    print(f"Saving data to cache at {WORKING_DIR}...")
    os.makedirs(WORKING_DIR, exist_ok=True)

    X_train.to_parquet(cache_files["X_train"])
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["train_ids"], train_ids)

    X_val.to_parquet(cache_files["X_val"])
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["val_ids"], val_ids)

    X_test.to_parquet(cache_files["X_test"])
    np.save(cache_files["test_ids"], test_ids)

    return X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids
