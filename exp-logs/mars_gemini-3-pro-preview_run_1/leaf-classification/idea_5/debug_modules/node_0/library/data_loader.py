import os
import numpy as np
import pandas as pd
from library.config import Config


def load_datasets(load_cached_data=True, sample_size=None):
    """
    Loads the training, validation, and test datasets.

    This function handles data ingestion from the metadata CSVs, extracts the
    relevant feature columns (margin, shape, texture), and separates targets
    and identifiers. It implements a caching mechanism using .npy files to
    speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from
                                 the cache directory defined in Config.
        sample_size (int, optional): If provided, returns a subset of the data
                                     (first N rows) for debugging purposes.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Training features (float32).
            - y_train (np.ndarray): Training labels (strings).
            - X_val (np.ndarray): Validation features (float32).
            - y_val (np.ndarray): Validation labels (strings).
            - X_test (np.ndarray): Test features (float32).
            - test_ids (np.ndarray): Test image IDs (int).
    """

    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(Config.CACHE_DIR, "X_val.npy"),
        "y_val": os.path.join(Config.CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
    }

    data_loaded = False

    # 1. Attempt to load from cache
    if load_cached_data:
        # Check if all cache files exist
        if all(os.path.exists(p) for p in cache_files.values()):
            print("Loading datasets from cache...")
            try:
                X_train = np.load(cache_files["X_train"])
                y_train = np.load(cache_files["y_train"], allow_pickle=True)
                X_val = np.load(cache_files["X_val"])
                y_val = np.load(cache_files["y_val"], allow_pickle=True)
                X_test = np.load(cache_files["X_test"])
                test_ids = np.load(cache_files["test_ids"])
                data_loaded = True
            except Exception as e:
                print(f"Error loading cache: {e}. Falling back to raw processing.")
                data_loaded = False
        else:
            print("Cache miss. Processing datasets from source...")

    # 2. Process from raw metadata if not loaded from cache
    if not data_loaded:
        print("Reading metadata CSVs...")
        df_train = pd.read_csv(Config.TRAIN_CSV)
        df_val = pd.read_csv(Config.VAL_CSV)
        df_test = pd.read_csv(Config.TEST_CSV)

        # Construct feature column names: margin_1..64, shape_1..64, texture_1..64
        feature_cols = []
        for group in Config.FEATURE_GROUPS:
            for i in range(1, Config.FEATURES_PER_GROUP + 1):
                feature_cols.append(f"{group}_{i}")

        print(f"Extracting {len(feature_cols)} features...")

        # Extract features and convert to float32
        X_train = df_train[feature_cols].values.astype(np.float32)
        y_train = df_train[Config.TARGET_COL].values

        X_val = df_val[feature_cols].values.astype(np.float32)
        y_val = df_val[Config.TARGET_COL].values

        X_test = df_test[feature_cols].values.astype(np.float32)
        test_ids = df_test[Config.ID_COL].values

        # Save to cache
        print(f"Caching datasets to {Config.CACHE_DIR}...")
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.save(cache_files["X_train"], X_train)
        np.save(cache_files["y_train"], y_train)
        np.save(cache_files["X_val"], X_val)
        np.save(cache_files["y_val"], y_val)
        np.save(cache_files["X_test"], X_test)
        np.save(cache_files["test_ids"], test_ids)

    # 3. Apply subsampling if requested (for debugging)
    if sample_size is not None:
        print(f"Subsampling datasets to {sample_size} samples.")
        X_train = X_train[:sample_size]
        y_train = y_train[:sample_size]
        X_val = X_val[:sample_size]
        y_val = y_val[:sample_size]
        X_test = X_test[:sample_size]
        test_ids = test_ids[:sample_size]

    return X_train, y_train, X_val, y_val, X_test, test_ids
