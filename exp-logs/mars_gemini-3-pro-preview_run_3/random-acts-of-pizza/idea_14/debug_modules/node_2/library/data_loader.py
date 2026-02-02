import os
import pandas as pd
import numpy as np
from library.config import Config


def load_and_preprocess_data(load_cached_data: bool = True):
    """
    Loads train, validation, and test datasets from metadata Parquet files.
    Applies initial preprocessing including leakage prevention and text standardization.
    Implements caching to speed up subsequent calls.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_cleaned.parquet")
    val_cache_path = os.path.join(cache_dir, "val_cleaned.parquet")
    test_cache_path = os.path.join(cache_dir, "test_cleaned.parquet")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            print("Loading cleaned data from cache...")
            try:
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print("Cache not found. Processing from scratch...")
    else:
        print("Ignoring cache. Processing from scratch...")

    # Load raw metadata
    print("Loading raw metadata...")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    def preprocess(df, is_test=False):
        # 1. Leakage Prevention: Drop retrieval-time features
        # Identify columns ending with defined suffixes
        cols_to_drop = [
            c
            for c in df.columns
            if any(c.endswith(suffix) for suffix in Config.LEAKAGE_SUFFIXES)
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # 2. Text Standardization
        # Use edit-aware text if available, fallback to raw text
        # We assign this to the Config.TEXT_COL so downstream components use the correct version consistently
        if Config.TEXT_EDIT_AWARE_COL in df.columns:
            # Fill NaNs in edit_aware with raw text (if raw text exists)
            if Config.TEXT_COL in df.columns:
                df[Config.TEXT_COL] = (
                    df[Config.TEXT_EDIT_AWARE_COL]
                    .fillna(df[Config.TEXT_COL])
                    .astype(str)
                )
            else:
                df[Config.TEXT_COL] = df[Config.TEXT_EDIT_AWARE_COL].astype(str)
        else:
            # If edit aware col is missing, ensure TEXT_COL is string
            if Config.TEXT_COL in df.columns:
                df[Config.TEXT_COL] = df[Config.TEXT_COL].astype(str)

        # Fill empty strings if any remain
        df[Config.TEXT_COL] = df[Config.TEXT_COL].fillna("")

        # 3. Type Casting
        # Ensure target is int if present
        if Config.TARGET_COL in df.columns:
            df[Config.TARGET_COL] = df[Config.TARGET_COL].astype(int)

        return df

    # Apply preprocessing
    print("Preprocessing data...")
    train_df = preprocess(train_df, is_test=False)
    val_df = preprocess(val_df, is_test=False)
    test_df = preprocess(test_df, is_test=True)

    # Save to cache
    print("Saving cleaned data to cache...")
    try:
        train_df.to_parquet(train_cache_path, index=False)
        val_df.to_parquet(val_cache_path, index=False)
        test_df.to_parquet(test_cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return train_df, val_df, test_df
