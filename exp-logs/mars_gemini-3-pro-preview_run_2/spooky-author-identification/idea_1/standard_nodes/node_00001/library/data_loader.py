import os
import pandas as pd
import numpy as np
from library.config import Config


def load_datasets(load_cached_data=True, debug=Config.DEBUG):
    """
    Loads the train, validation, and test datasets.
    Implements caching using parquet files to speed up subsequent loads.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
        debug (bool): If True, loads a smaller subset of the data for debugging.

    Returns:
        tuple: A tuple containing:
            (X_train, y_train, train_ids): text (Series), author (Series), id (Series)
            (X_val, y_val, val_ids): text (Series), author (Series), id (Series)
            (X_test, test_ids): text (Series), id (Series)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache filenames based on debug status to avoid mixing full/subset data
    suffix = "_debug" if debug else ""
    cache_train_path = os.path.join(Config.WORKING_DIR, f"train{suffix}.parquet")
    cache_val_path = os.path.join(Config.WORKING_DIR, f"val{suffix}.parquet")
    cache_test_path = os.path.join(Config.WORKING_DIR, f"test{suffix}.parquet")

    train_df = None
    val_df = None
    test_df = None

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):
            print("Loading datasets from cache...")
            try:
                train_df = pd.read_parquet(cache_train_path)
                val_df = pd.read_parquet(cache_val_path)
                test_df = pd.read_parquet(cache_test_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from source.")
                train_df = None

    # 2. Load from source if not loaded from cache
    if train_df is None:
        print("Loading datasets from metadata source...")

        # Read CSVs
        train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
        val_df = pd.read_csv(Config.VAL_DATA_PATH)
        test_df = pd.read_csv(Config.TEST_DATA_PATH)

        # Apply Debug Sampling
        if debug:
            print(
                f"Debug mode enabled. Sampling {Config.DEBUG_SAMPLES} rows per dataset."
            )
            # Sample training data
            if len(train_df) > Config.DEBUG_SAMPLES:
                train_df = train_df.sample(
                    n=Config.DEBUG_SAMPLES, random_state=Config.SEED
                ).reset_index(drop=True)

            # Sample validation data (proportionally smaller or same cap, here we just cap)
            if len(val_df) > Config.DEBUG_SAMPLES:
                val_df = val_df.sample(
                    n=Config.DEBUG_SAMPLES, random_state=Config.SEED
                ).reset_index(drop=True)

            # Sample test data
            if len(test_df) > Config.DEBUG_SAMPLES:
                test_df = test_df.sample(
                    n=Config.DEBUG_SAMPLES, random_state=Config.SEED
                ).reset_index(drop=True)

        # Save to cache
        print(f"Saving datasets to cache at {Config.WORKING_DIR}...")
        train_df.to_parquet(cache_train_path, index=False)
        val_df.to_parquet(cache_val_path, index=False)
        test_df.to_parquet(cache_test_path, index=False)

    # 3. Separate features, labels, and IDs
    print(
        f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
    )

    # Training data
    X_train = train_df[Config.TEXT_COL]
    y_train = train_df[Config.TARGET_COL]
    train_ids = train_df[Config.ID_COL]

    # Validation data
    X_val = val_df[Config.TEXT_COL]
    y_val = val_df[Config.TARGET_COL]
    val_ids = val_df[Config.ID_COL]

    # Test data (no target)
    X_test = test_df[Config.TEXT_COL]
    test_ids = test_df[Config.ID_COL]

    return (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids)
