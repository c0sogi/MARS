import os
import pandas as pd
import numpy as np
from library.config import TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, WORKING_DIR


def load_data(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.
    Implements caching using Parquet files to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache file paths
    train_cache_path = os.path.join(WORKING_DIR, "train_base.parquet")
    val_cache_path = os.path.join(WORKING_DIR, "val_base.parquet")
    test_cache_path = os.path.join(WORKING_DIR, "test_base.parquet")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):

            print("Loading data from cache...")
            try:
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Proceeding to load from source.")
        else:
            print("Cache files not found. Loading from source...")
    else:
        print("Ignoring cache. Loading from source...")

    # Load from metadata CSVs
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train data not found at {TRAIN_DATA_PATH}")

    print(f"Reading train data from {TRAIN_DATA_PATH}...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    print(f"Reading validation data from {VAL_DATA_PATH}...")
    val_df = pd.read_csv(VAL_DATA_PATH)

    print(f"Reading test data from {TEST_DATA_PATH}...")
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Basic Cleaning and Type Conversion

    # 1. Target Variable Conversion (Boolean -> Int)
    target_col = "requester_received_pizza"
    if target_col in train_df.columns:
        train_df[target_col] = train_df[target_col].astype(int)
    if target_col in val_df.columns:
        val_df[target_col] = val_df[target_col].astype(int)

    # 2. Boolean Feature Conversion
    # 'post_was_edited' is often boolean or 0/1. Ensure it is integer 0/1.
    bool_cols = ["post_was_edited"]

    for df in [train_df, val_df, test_df]:
        for col in bool_cols:
            if col in df.columns:
                # Handle potential string representations of boolean or NaNs
                # Map string "True"/"False" to 1/0 before casting
                df[col] = df[col].replace(
                    {"True": 1, "False": 0, "true": 1, "false": 0}
                )
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 3. Ensure timestamps are numeric
    time_cols = ["unix_timestamp_of_request", "unix_timestamp_of_request_utc"]
    for df in [train_df, val_df, test_df]:
        for col in time_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Save to cache
    print("Saving processed data to cache...")
    try:
        train_df.to_parquet(train_cache_path, index=False)
        val_df.to_parquet(val_cache_path, index=False)
        test_df.to_parquet(test_cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save to cache: {e}")

    return train_df, val_df, test_df
