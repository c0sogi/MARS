import os
import pandas as pd
import numpy as np
from library import config


def enforce_leakage_prevention(train_df, val_df, test_df):
    """
    Restricts columns to the intersection of training and test sets to prevent data leakage.
    Preserves the target column in training and validation sets.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.

    Returns:
        tuple: Filtered (train_df, val_df, test_df).
    """
    target_col = "requester_received_pizza"

    # Identify columns present in both train and test
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # The intersection of features available at inference time
    common_cols = train_cols.intersection(test_cols)

    # Ensure target is not in common_cols (it shouldn't be in test, but remove to be safe)
    if target_col in common_cols:
        common_cols.remove(target_col)

    # Convert to sorted list for deterministic column order
    common_cols_list = sorted(list(common_cols))

    # Columns to keep for training/validation (Features + Target)
    train_keep_cols = common_cols_list + [target_col]

    # Columns to keep for test (Features only)
    test_keep_cols = common_cols_list

    # Filter DataFrames
    # Use .copy() to ensure we have clean dataframes and avoid SettingWithCopyWarning
    train_filtered = train_df[train_keep_cols].copy()
    val_filtered = val_df[train_keep_cols].copy()
    test_filtered = test_df[test_keep_cols].copy()

    print(f"Leakage Prevention Applied:")
    print(f"  - Common Feature Columns: {len(common_cols_list)}")
    print(f"  - Train Shape: {train_filtered.shape}")
    print(f"  - Val Shape: {val_filtered.shape}")
    print(f"  - Test Shape: {test_filtered.shape}")

    return train_filtered, val_filtered, test_filtered


def load_dataset(load_cached_data=True, debug=False):
    """
    Loads the dataset, applies leakage prevention, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed parquet files.
        debug (bool): If True, subsamples the data for rapid testing.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths from config
    cache_train_path = config.CACHE_PATHS["train_processed"]
    cache_val_path = config.CACHE_PATHS["val_processed"]
    cache_test_path = config.CACHE_PATHS["test_processed"]

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):

            print("Loading datasets from cache...")
            train_df = pd.read_parquet(cache_train_path)
            val_df = pd.read_parquet(cache_val_path)
            test_df = pd.read_parquet(cache_test_path)

            # Validate cache against configuration (Cite debug_lesson_1)
            # If not in debug mode, but cache is small, it's a stale debug cache.
            if not debug and len(train_df) <= config.DEBUG_SAMPLE_SIZE:
                print(
                    f"Detected stale debug cache (Train size: {len(train_df)}). Reloading from raw source..."
                )
            else:
                # If debug is requested, slice the cached data
                if debug:
                    print(
                        f"Debug mode: Subsampling to {config.DEBUG_SAMPLE_SIZE} samples."
                    )
                    train_df = train_df.head(config.DEBUG_SAMPLE_SIZE)
                    val_df = val_df.head(config.DEBUG_SAMPLE_SIZE)
                    test_df = test_df.head(config.DEBUG_SAMPLE_SIZE)

                return train_df, val_df, test_df
        else:
            print("Cache not found. Processing from scratch...")
    else:
        print("Ignoring cache. Processing from scratch...")

    # Load raw metadata CSVs
    print("Loading raw metadata CSVs...")
    if not os.path.exists(config.TRAIN_PATH):
        raise FileNotFoundError(f"Train file not found at {config.TRAIN_PATH}")
    if not os.path.exists(config.VAL_PATH):
        raise FileNotFoundError(f"Val file not found at {config.VAL_PATH}")
    if not os.path.exists(config.TEST_PATH):
        raise FileNotFoundError(f"Test file not found at {config.TEST_PATH}")

    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    # Apply Debug Subsampling BEFORE processing to save time if debugging
    if debug:
        print(f"Debug mode: Subsampling to {config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.head(config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(config.DEBUG_SAMPLE_SIZE)

    # Apply Leakage Prevention
    train_df, val_df, test_df = enforce_leakage_prevention(train_df, val_df, test_df)

    # Save to Cache
    # We only save if NOT in debug mode to prevent overwriting full cache with subsampled data
    if not debug:
        print("Saving processed datasets to cache...")
        train_df.to_parquet(cache_train_path, index=False)
        val_df.to_parquet(cache_val_path, index=False)
        test_df.to_parquet(cache_test_path, index=False)
    else:
        print("Debug mode: Skipping cache save to prevent overwriting full dataset.")

    return train_df, val_df, test_df
