import os
import numpy as np
import pandas as pd
import library.config as config


def load_data(load_cached_data=True):
    """
    Loads, processes, and caches the dataset.

    Enforces Alphanumeric Column Ordering and float64 precision.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
    """
    # Define cache file paths
    cache_X_train = os.path.join(config.CACHE_DIR, "X_train.parquet")
    cache_y_train = os.path.join(config.CACHE_DIR, "y_train.npy")
    cache_X_val = os.path.join(config.CACHE_DIR, "X_val.parquet")
    cache_y_val = os.path.join(config.CACHE_DIR, "y_val.npy")
    cache_X_test = os.path.join(config.CACHE_DIR, "X_test.parquet")
    cache_test_ids = os.path.join(config.CACHE_DIR, "test_ids.npy")
    cache_classes = os.path.join(config.CACHE_DIR, "classes.npy")

    # Check if cache exists
    files_exist = all(
        os.path.exists(f)
        for f in [
            cache_X_train,
            cache_y_train,
            cache_X_val,
            cache_y_val,
            cache_X_test,
            cache_test_ids,
            cache_classes,
        ]
    )

    if load_cached_data and files_exist:
        print("Loading data from cache...")
        X_train = pd.read_parquet(cache_X_train)
        y_train = np.load(cache_y_train, allow_pickle=True)
        X_val = pd.read_parquet(cache_X_val)
        y_val = np.load(cache_y_val, allow_pickle=True)
        X_test = pd.read_parquet(cache_X_test)
        test_ids = np.load(cache_test_ids, allow_pickle=True)
        classes = np.load(cache_classes, allow_pickle=True)

        return X_train, y_train, X_val, y_val, X_test, test_ids, classes

    print("Processing raw data from metadata...")

    # Load metadata CSVs
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_val = pd.read_csv(config.VAL_FILE)
    df_test = pd.read_csv(config.TEST_FILE)

    # Identify feature columns
    # We look for columns starting with the defined prefixes
    all_cols = df_train.columns.tolist()
    feature_cols = [
        c
        for c in all_cols
        if any(c.startswith(prefix) for prefix in config.FEATURE_PREFIXES)
    ]

    # STRICTLY enforce Alphanumeric Column Ordering
    # Standard python sorted() performs lexicographical sort:
    # e.g. ['margin_1', 'margin_10', 'margin_2']
    feature_cols = sorted(feature_cols)

    # Extract Features and cast to float64
    X_train = df_train[feature_cols].astype(config.DTYPE)
    X_val = df_val[feature_cols].astype(config.DTYPE)
    X_test = df_test[feature_cols].astype(config.DTYPE)

    # Extract Targets and IDs
    y_train = df_train["species"].values
    y_val = df_val["species"].values
    test_ids = df_test["id"].values

    # Extract Classes (sorted)
    classes = np.unique(y_train)

    # Save to cache
    print(f"Saving processed data to {config.CACHE_DIR}...")
    X_train.to_parquet(cache_X_train)
    np.save(cache_y_train, y_train)
    X_val.to_parquet(cache_X_val)
    np.save(cache_y_val, y_val)
    X_test.to_parquet(cache_X_test)
    np.save(cache_test_ids, test_ids)
    np.save(cache_classes, classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
