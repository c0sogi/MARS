import os
import pandas as pd
import numpy as np
from library import config


def load_datasets(load_cached_data=True, debug_size=config.DEBUG_SAMPLE_SIZE):
    """
    Loads the train, validation, and test datasets. Implements caching to Parquet
    to optimize loading times.

    Args:
        load_cached_data (bool): If True, attempts to load from Parquet cache.
                                 If False or cache missing, loads from CSV and updates cache.
        debug_size (int or None): If set, limits the number of rows loaded for debugging.

    Returns:
        tuple: (df_train, df_val, df_test) loaded pandas DataFrames.
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    cache_train = os.path.join(config.CACHE_DIR, "train.parquet")
    cache_val = os.path.join(config.CACHE_DIR, "val.parquet")
    cache_test = os.path.join(config.CACHE_DIR, "test.parquet")

    # Determine if we should load from cache
    cache_exists = (
        os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    )

    if load_cached_data and cache_exists:
        print("Loading datasets from Parquet cache...")
        df_train = pd.read_parquet(cache_train)
        df_val = pd.read_parquet(cache_val)
        df_test = pd.read_parquet(cache_test)
    else:
        print("Loading datasets from Metadata CSVs...")
        # Load raw CSVs
        df_train = pd.read_csv(config.TRAIN_DATA_PATH)
        df_val = pd.read_csv(config.VAL_DATA_PATH)
        df_test = pd.read_csv(config.TEST_DATA_PATH)

        # Enforce float precision on feature columns before caching
        # We identify feature columns by excluding metadata columns
        exclude_cols = {
            config.ID_COLUMN,
            config.TARGET_COLUMN,
            "file_path",
            "full_path",
        }

        for df in [df_train, df_val, df_test]:
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            # Convert feature columns to specified precision
            df[feature_cols] = df[feature_cols].astype(config.FLOAT_PRECISION)

        # Save to cache
        print(f"Saving datasets to cache at {config.CACHE_DIR}...")
        df_train.to_parquet(cache_train)
        df_val.to_parquet(cache_val)
        df_test.to_parquet(cache_test)

    # Apply debug slicing if requested
    # We slice after loading to ensure the cache always contains the full dataset
    if debug_size is not None:
        print(f"Debug Mode: Slicing datasets to {debug_size} samples.")
        df_train = df_train.iloc[:debug_size].copy()

        # Filter validation set to strictly match training classes to prevent LabelEncoder errors
        # Cite debug_lesson_1: Filter Classes, Don't Just Slice Rows, for High-Cardinality Debugging
        train_classes = df_train[config.TARGET_COLUMN].unique()
        df_val = df_val[df_val[config.TARGET_COLUMN].isin(train_classes)].copy()

        # Slice validation to debug size if it's still larger
        if len(df_val) > debug_size:
            df_val = df_val.iloc[:debug_size].copy()

        # Test set is usually small, but we slice it too if requested to maintain consistency
        df_test = df_test.iloc[:debug_size].copy()

    return df_train, df_val, df_test
