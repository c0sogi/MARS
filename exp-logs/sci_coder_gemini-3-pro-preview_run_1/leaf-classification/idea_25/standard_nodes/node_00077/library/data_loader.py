import pandas as pd
import numpy as np
import os
from library import config


def load_dataset(load_cached_data=True, debug_size=None):
    """
    Loads the dataset using the pre-defined metadata splits.

    Implements a caching mechanism:
    - Checks ./working/idea_25/ for .parquet (features) and .npy (targets) files.
    - If found and load_cached_data=True, loads them.
    - Else, reads from ./metadata/, processes features to float64, saves to cache, and returns.

    Args:
        load_cached_data (bool): Whether to attempt loading from the local cache.
        debug_size (int, optional): If set, returns only the first N samples for debugging.

    Returns:
        X_train (pd.DataFrame): Training features (float64)
        y_train (np.ndarray): Training labels
        X_val (pd.DataFrame): Validation features (float64)
        y_val (np.ndarray): Validation labels
        X_test (pd.DataFrame): Test features (float64)
        test_ids (np.ndarray): Test IDs
    """

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_paths = {
        "X_train": os.path.join(config.WORKING_DIR, "X_train.parquet"),
        "y_train": os.path.join(config.WORKING_DIR, "y_train.npy"),
        "X_val": os.path.join(config.WORKING_DIR, "X_val.parquet"),
        "y_val": os.path.join(config.WORKING_DIR, "y_val.npy"),
        "X_test": os.path.join(config.WORKING_DIR, "X_test.parquet"),
        "test_ids": os.path.join(config.WORKING_DIR, "test_ids.npy"),
    }

    data_loaded = False

    # Attempt to load from cache
    if load_cached_data:
        if all(os.path.exists(p) for p in cache_paths.values()):
            print("Loading dataset from cache...")
            try:
                X_train = pd.read_parquet(cache_paths["X_train"])
                y_train = np.load(cache_paths["y_train"], allow_pickle=True)
                X_val = pd.read_parquet(cache_paths["X_val"])
                y_val = np.load(cache_paths["y_val"], allow_pickle=True)
                X_test = pd.read_parquet(cache_paths["X_test"])
                test_ids = np.load(cache_paths["test_ids"], allow_pickle=True)
                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from metadata...")
        else:
            print("Cache missing or incomplete. Loading from metadata...")

    # Load from metadata if cache not used or failed
    if not data_loaded:
        print("Processing raw data from metadata...")

        # Paths to metadata
        meta_train = os.path.join(config.METADATA_DIR, "train.csv")
        meta_val = os.path.join(config.METADATA_DIR, "val.csv")
        meta_test = os.path.join(config.METADATA_DIR, "test.csv")

        # Read CSVs
        df_train = pd.read_csv(meta_train)
        df_val = pd.read_csv(meta_val)
        df_test = pd.read_csv(meta_test)

        # Select features and cast to float64
        # We enforce the specific feature columns defined in config to ensure deterministic order
        X_train = df_train[config.FEATURE_COLUMNS].astype(config.FLOAT_PRECISION)
        X_val = df_val[config.FEATURE_COLUMNS].astype(config.FLOAT_PRECISION)
        X_test = df_test[config.FEATURE_COLUMNS].astype(config.FLOAT_PRECISION)

        # Extract targets and IDs
        y_train = df_train[config.TARGET_COLUMN].values
        y_val = df_val[config.TARGET_COLUMN].values
        test_ids = df_test[config.ID_COLUMN].values

        # Save to cache for future runs
        print(f"Saving processed dataset to {config.WORKING_DIR}...")
        X_train.to_parquet(cache_paths["X_train"])
        np.save(cache_paths["y_train"], y_train)

        X_val.to_parquet(cache_paths["X_val"])
        np.save(cache_paths["y_val"], y_val)

        X_test.to_parquet(cache_paths["X_test"])
        np.save(cache_paths["test_ids"], test_ids)

    # Handle debugging subsample
    if debug_size is not None and debug_size > 0:
        print(f"Subsampling dataset to {debug_size} samples for debugging.")
        X_train = X_train.iloc[:debug_size]
        y_train = y_train[:debug_size]
        X_val = X_val.iloc[:debug_size]
        y_val = y_val[:debug_size]
        X_test = X_test.iloc[:debug_size]
        test_ids = test_ids[:debug_size]

    return X_train, y_train, X_val, y_val, X_test, test_ids
