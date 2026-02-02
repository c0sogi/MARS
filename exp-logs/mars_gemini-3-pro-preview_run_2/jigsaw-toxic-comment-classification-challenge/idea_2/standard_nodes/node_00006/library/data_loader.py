import os
import pandas as pd
from library.config import Config


def load_datasets(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.
    Implements caching using Parquet to speed up subsequent loads.

    Args:
        load_cached_data (bool): If True, attempts to load from Parquet cache.
                                 If False or cache missing, loads from CSV and updates cache.

    Returns:
        tuple: (train_df, val_df, test_df) containing the loaded pandas DataFrames.
    """
    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test.parquet")

    train_df = None
    val_df = None
    test_df = None

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading datasets from Parquet cache...")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
            except Exception as e:
                print(f"Failed to load cache: {e}. Reloading from CSV.")
                train_df = None  # Force reload

    # 2. If not loaded (cache miss, forced reload, or load_cached_data=False), load from CSV
    if train_df is None:
        print("Loading datasets from CSV metadata...")
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)
        test_df = pd.read_csv(Config.TEST_CSV)

        # Preprocessing: Handle missing text
        # It's crucial to fill NaNs before saving to ensure consistency
        print("Preprocessing: Filling missing text values...")
        train_df[Config.TEXT_COL] = train_df[Config.TEXT_COL].fillna("")
        val_df[Config.TEXT_COL] = val_df[Config.TEXT_COL].fillna("")
        test_df[Config.TEXT_COL] = test_df[Config.TEXT_COL].fillna("")

        # Save to cache
        print(f"Saving datasets to Parquet cache at {Config.WORKING_DIR}...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    # 3. Handle Debug Mode
    # We apply debug slicing AFTER loading (whether from cache or CSV)
    # so that the cache always contains the full dataset.
    if Config.DEBUG:
        print(
            f"DEBUG mode enabled. Slicing datasets to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

    return train_df, val_df, test_df
