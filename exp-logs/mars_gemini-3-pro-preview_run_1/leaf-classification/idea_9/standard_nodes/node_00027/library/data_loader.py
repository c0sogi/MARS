import os
import pandas as pd
import numpy as np
from library import config


def load_data(load_cached_data=True, sample_size=None):
    """
    Loads the training, validation, and test data.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays from cache.
        sample_size (int, optional): If provided, limits the number of samples for debugging.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
            X_train, X_val, X_test: Feature matrices (numpy arrays).
            y_train, y_val: Target vectors (numpy arrays).
            test_ids: Array of IDs for the test set.
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(config.CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(config.CACHE_DIR, "X_val.npy"),
        "y_val": os.path.join(config.CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(config.CACHE_DIR, "X_test.npy"),
        "test_ids": os.path.join(config.CACHE_DIR, "test_ids.npy"),
    }

    # Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print(f"Loading data from cache: {config.CACHE_DIR}")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"], allow_pickle=True)
            X_val = np.load(cache_files["X_val"])
            y_val = np.load(cache_files["y_val"], allow_pickle=True)
            X_test = np.load(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"])

            # Apply sampling if requested on cached data
            if sample_size is not None:
                X_train = X_train[:sample_size]
                y_train = y_train[:sample_size]
                X_val = X_val[:sample_size]
                y_val = y_val[:sample_size]
                # We generally don't sample test set in a way that breaks submission,
                # but for strict debugging consistency we can.
                # However, usually test set size is fixed. We'll leave test set alone
                # or sample it if strictly necessary for pipeline checks.
                # For this implementation, we only sample train/val to speed up training debug.

            return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Loading data from raw CSV files...")

    # Load CSVs
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(config.VAL_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH)

    # Apply sampling if requested
    if sample_size is not None:
        print(f"Sampling {sample_size} rows from training and validation sets.")
        df_train = df_train.head(sample_size)
        df_val = df_val.head(sample_size)
        # df_test = df_test.head(sample_size) # Keep test full for submission structure validity

    # Identify feature columns based on prefixes
    # We scan columns of df_train to find those starting with defined prefixes
    feature_cols = [
        col
        for col in df_train.columns
        if any(col.startswith(prefix) for prefix in config.FEATURE_PREFIXES)
    ]

    # Sort feature columns to ensure consistent order
    feature_cols.sort()

    print(f"Identified {len(feature_cols)} feature columns.")

    # Extract Features
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    # Extract Targets
    y_train = df_train[config.TARGET_COL].values
    y_val = df_val[config.TARGET_COL].values

    # Extract Test IDs
    test_ids = df_test[config.ID_COL].values

    # Save to cache
    print(f"Saving processed data to cache: {config.CACHE_DIR}")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val"], X_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids
