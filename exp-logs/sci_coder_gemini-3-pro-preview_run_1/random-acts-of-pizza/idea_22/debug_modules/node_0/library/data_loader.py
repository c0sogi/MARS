import os
import pandas as pd
import numpy as np
from library.config import Config


def load_dataset(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.
    Implements caching using Parquet files to speed up subsequent runs.
    Handles DEBUG mode by sampling the datasets.

    Args:
        load_cached_data (bool): If True, attempts to load from cached Parquet files.
                                 If False or cache missing, loads from original CSVs.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_train_path = os.path.join(Config.WORKING_DIR, "train_cache.parquet")
    cache_val_path = os.path.join(Config.WORKING_DIR, "val_cache.parquet")
    cache_test_path = os.path.join(Config.WORKING_DIR, "test_cache.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    )

    if load_cached_data and cache_exists:
        print("Loading datasets from cache...")
        train_df = pd.read_parquet(cache_train_path)
        val_df = pd.read_parquet(cache_val_path)
        test_df = pd.read_parquet(cache_test_path)
    else:
        print("Loading datasets from metadata CSVs...")
        # Load from CSVs specified in Config
        train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
        val_df = pd.read_csv(Config.VAL_DATA_PATH)
        test_df = pd.read_csv(Config.TEST_DATA_PATH)

        # Ensure boolean target is properly typed if present
        target_col = "requester_received_pizza"
        if target_col in train_df.columns:
            train_df[target_col] = train_df[target_col].astype(bool)
        if target_col in val_df.columns:
            val_df[target_col] = val_df[target_col].astype(bool)

        # Save to cache
        print(f"Saving datasets to cache at {Config.WORKING_DIR}...")
        train_df.to_parquet(cache_train_path, index=False)
        val_df.to_parquet(cache_val_path, index=False)
        test_df.to_parquet(cache_test_path, index=False)

    # Handle DEBUG mode (downsampling)
    if Config.DEBUG:
        print(f"DEBUG mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        # Use a fixed seed for reproducibility in debug sampling
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE),
            random_state=Config.RANDOM_SEED,
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE),
            random_state=Config.RANDOM_SEED,
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE),
            random_state=Config.RANDOM_SEED,
        ).reset_index(drop=True)

    return train_df, val_df, test_df


def get_common_columns(train_df, test_df, target_col="requester_received_pizza"):
    """
    Identifies the intersection of columns between train and test dataframes
    to prevent data leakage. Explicitly excludes target, identifiers, and
    known leakage columns.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        target_col (str): Name of the target variable column.

    Returns:
        list: A sorted list of column names safe for feature usage.
    """
    # Columns to explicitly exclude (identifiers, leakage, targets)
    excluded_cols = {
        target_col,
        "request_id",
        "giver_username_if_known",
        "source_file",
        "requester_username",  # Often high cardinality/identifier
        "request_text",  # Raw text is usually processed into features, not used directly
        "request_title",
        "request_text_edit_aware",
    }

    # Find intersection
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)

    # Remove excluded columns
    feature_cols = [col for col in common_cols if col not in excluded_cols]

    # Sort for deterministic order
    feature_cols.sort()

    return feature_cols
