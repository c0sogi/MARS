import os
import pandas as pd
import numpy as np
from library import config


def load_metadata_splits(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    Implements a caching mechanism using Parquet files to speed up loading
    and preserve data types in subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from the processed
                                 parquet cache first. If False or cache missing,
                                 loads from raw metadata CSVs and updates cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = config.TRAIN_PROCESSED_PATH
    val_cache = config.VAL_PROCESSED_PATH
    test_cache = config.TEST_PROCESSED_PATH

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print(f"Loading datasets from cache: {config.WORKING_DIR}")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Falling back to raw CSVs.")
        else:
            print("Cache files not found. Loading from raw CSVs.")
    else:
        print("Cache loading disabled. Loading from raw CSVs.")

    # Load from raw metadata CSVs
    print(f"Reading train data from {config.TRAIN_PATH}")
    train_df = pd.read_csv(config.TRAIN_PATH)

    print(f"Reading validation data from {config.VAL_PATH}")
    val_df = pd.read_csv(config.VAL_PATH)

    print(f"Reading test data from {config.TEST_PATH}")
    test_df = pd.read_csv(config.TEST_PATH)

    # Enforce Data Types
    # 1. Target Column (Boolean/Int)
    if config.TARGET_COL in train_df.columns:
        train_df[config.TARGET_COL] = train_df[config.TARGET_COL].astype(int)
    if config.TARGET_COL in val_df.columns:
        val_df[config.TARGET_COL] = val_df[config.TARGET_COL].astype(int)

    # 2. Text Columns (String) - Handle NaNs
    text_cols = [config.TEXT_COL, "request_title", "request_text"]

    for df in [train_df, val_df, test_df]:
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

    # Save to Cache (Parquet)
    print(f"Saving processed datasets to cache: {config.WORKING_DIR}")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df
