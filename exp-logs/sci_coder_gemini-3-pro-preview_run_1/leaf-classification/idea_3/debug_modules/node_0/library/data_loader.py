import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library import config


def load_data(load_cached_data=True):
    """
    Loads, processes, and returns the training, validation, and test datasets.
    Implements caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays
                                 from the working directory.

    Returns:
        tuple: A tuple containing:
            - X_train (np.array): Training features.
            - y_train (np.array): Encoded training labels.
            - X_val (np.array): Validation features.
            - y_val (np.array): Encoded validation labels.
            - X_test (np.array): Test features.
            - test_ids (np.array): IDs for the test images.
            - classes (np.array): Original class names corresponding to encoded labels.
    """
    # Define cache file paths based on config
    cache_files = {
        "X_train": config.X_TRAIN_CACHE,
        "y_train": config.Y_TRAIN_CACHE,
        "X_val": config.X_VAL_CACHE,
        "y_val": config.Y_VAL_CACHE,
        "X_test": config.X_TEST_CACHE,
        "test_ids": config.TEST_IDS_CACHE,
        "classes": config.CLASSES_CACHE,
    }

    # 1. Check Cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"])
            X_val = np.load(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"])
            X_test = np.load(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"])
            classes = np.load(cache_files["classes"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, test_ids, classes
        else:
            print("Cache miss. Processing data from scratch...")

    # 2. Load Metadata CSVs
    print("Loading metadata CSVs...")
    df_train = pd.read_csv(config.TRAIN_CSV)
    df_val = pd.read_csv(config.VAL_CSV)
    df_test = pd.read_csv(config.TEST_CSV)

    # 3. Identify Feature Columns
    # Exclude non-feature columns defined in config
    feature_cols = [c for c in df_train.columns if c not in config.EXCLUDE_COLS]

    # Ensure consistent column ordering
    feature_cols.sort()

    # 4. Extract Features and Targets
    print(f"Extracting {len(feature_cols)} features...")

    # Train
    X_train = df_train[feature_cols].values.astype(np.float64)
    y_train_raw = df_train[config.TARGET_COL].values

    # Validation
    X_val = df_val[feature_cols].values.astype(np.float64)
    y_val_raw = df_val[config.TARGET_COL].values

    # Test
    X_test = df_test[feature_cols].values.astype(np.float64)
    test_ids = df_test[config.ID_COL].values

    # 5. Encode Targets
    print("Encoding target labels...")
    le = LabelEncoder()
    # Fit on training labels
    y_train = le.fit_transform(y_train_raw)
    # Transform validation labels
    y_val = le.transform(y_val_raw)
    classes = le.classes_

    # 6. Save to Cache
    print("Saving processed data to cache...")
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
