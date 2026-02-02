import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    ALL_FEATURES,
    TARGET_COL,
    ID_COL,
    FLOAT_PRECISION,
    CACHE_TRAIN_X,
    CACHE_TRAIN_Y,
    CACHE_VAL_X,
    CACHE_VAL_Y,
    CACHE_TEST_X,
    CACHE_TEST_IDS,
    CACHE_CLASSES,
    WORKING_DIR,
)


def load_datasets(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.
    Implements caching to speed up subsequent runs.
    Enforces float64 precision for feature matrices.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
            - X_train (np.ndarray): Training features (float64).
            - y_train (np.ndarray): Training labels (strings).
            - X_val (np.ndarray): Validation features (float64).
            - y_val (np.ndarray): Validation labels (strings).
            - X_test (np.ndarray): Test features (float64).
            - test_ids (np.ndarray): Test image IDs.
            - classes (np.ndarray): Sorted unique class names.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Check if cached files exist
    cache_files = [
        CACHE_TRAIN_X,
        CACHE_TRAIN_Y,
        CACHE_VAL_X,
        CACHE_VAL_Y,
        CACHE_TEST_X,
        CACHE_TEST_IDS,
        CACHE_CLASSES,
    ]

    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print("Loading datasets from cache...")
        X_train = np.load(CACHE_TRAIN_X)
        y_train = np.load(CACHE_TRAIN_Y, allow_pickle=True)
        X_val = np.load(CACHE_VAL_X)
        y_val = np.load(CACHE_VAL_Y, allow_pickle=True)
        X_test = np.load(CACHE_TEST_X)
        test_ids = np.load(CACHE_TEST_IDS)
        classes = np.load(CACHE_CLASSES, allow_pickle=True)

        return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    print("Loading datasets from metadata CSVs...")

    # Load CSVs
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_val = pd.read_csv(VAL_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # Extract Features and enforce float64 precision
    # We use the hardcoded feature list to ensure deterministic column ordering
    X_train = df_train[ALL_FEATURES].values.astype(FLOAT_PRECISION)
    X_val = df_val[ALL_FEATURES].values.astype(FLOAT_PRECISION)
    X_test = df_test[ALL_FEATURES].values.astype(FLOAT_PRECISION)

    # Extract Labels
    y_train = df_train[TARGET_COL].values
    y_val = df_val[TARGET_COL].values

    # Extract IDs for submission
    test_ids = df_test[ID_COL].values

    # Extract Classes (sorted alphabetically)
    classes = np.unique(y_train)
    classes.sort()

    # Save to cache
    print("Saving processed datasets to cache...")
    np.save(CACHE_TRAIN_X, X_train)
    np.save(CACHE_TRAIN_Y, y_train)
    np.save(CACHE_VAL_X, X_val)
    np.save(CACHE_VAL_Y, y_val)
    np.save(CACHE_TEST_X, X_test)
    np.save(CACHE_TEST_IDS, test_ids)
    np.save(CACHE_CLASSES, classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
