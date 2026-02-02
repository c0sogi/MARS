import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    get_alphanumeric_feature_order,
)


def load_datasets(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.

    Enforces float64 precision and alphanumeric feature ordering to ensure
    consistent memory layout and numerical stability for the exact Cholesky solver.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data
                                 from the working directory cache.

    Returns:
        X_train (pd.DataFrame): Training features (float64, ordered).
        y_train (np.ndarray): Training labels.
        X_val (pd.DataFrame): Validation features (float64, ordered).
        y_val (np.ndarray): Validation labels.
        X_test (pd.DataFrame): Test features (float64, ordered).
        test_ids (np.ndarray): Test image IDs.
        classes (np.ndarray): Sorted list of unique species names.
    """
    # Define cache file paths
    cache_paths = {
        "X_train": os.path.join(WORKING_DIR, "X_train.parquet"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val.parquet"),
        "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.parquet"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
        "classes": os.path.join(WORKING_DIR, "classes.npy"),
    }

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_paths.values())
        if all_exist:
            print("Loading datasets from cache...")
            X_train = pd.read_parquet(cache_paths["X_train"])
            y_train = np.load(cache_paths["y_train"], allow_pickle=True)
            X_val = pd.read_parquet(cache_paths["X_val"])
            y_val = np.load(cache_paths["y_val"], allow_pickle=True)
            X_test = pd.read_parquet(cache_paths["X_test"])
            test_ids = np.load(cache_paths["test_ids"], allow_pickle=True)
            classes = np.load(cache_paths["classes"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, test_ids, classes
        else:
            print("Cache miss or partial cache found. Reloading from source...")

    # Load raw metadata
    print("Loading datasets from metadata CSVs...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # Enforce Alphanumeric Feature Ordering
    # This aligns the memory layout with the high-performance baseline
    feature_cols = get_alphanumeric_feature_order()

    # Process Training Data
    # Enforce float64 immediately to prevent precision loss
    X_train = df_train[feature_cols].astype("float64")
    y_train = df_train["species"].values

    # Process Validation Data
    X_val = df_val[feature_cols].astype("float64")
    y_val = df_val["species"].values

    # Process Test Data
    X_test = df_test[feature_cols].astype("float64")
    test_ids = df_test["id"].values

    # Extract and Sort Classes
    # Combining train and val to ensure all potential classes are captured,
    # though stratification usually handles this.
    classes = np.unique(np.concatenate([y_train, y_val]))
    classes.sort()

    # Save to Cache
    print("Saving processed datasets to cache...")
    X_train.to_parquet(cache_paths["X_train"])
    np.save(cache_paths["y_train"], y_train)

    X_val.to_parquet(cache_paths["X_val"])
    np.save(cache_paths["y_val"], y_val)

    X_test.to_parquet(cache_paths["X_test"])
    np.save(cache_paths["test_ids"], test_ids)

    np.save(cache_paths["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
