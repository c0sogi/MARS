import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    FLOAT_PRECISION,
    get_feature_columns,
)
from library.utils import alphanumeric_sort


def load_data(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    Implements a caching mechanism using Parquet files to store processed
    DataFrames with correct float64 precision.

    Args:
        load_cached_data (bool): If True, attempts to load from the cache directory.
                                 If False or cache miss, loads from metadata CSVs,
                                 processes types, and saves to cache.

    Returns:
        tuple: (df_train, df_val, df_test) as pandas DataFrames.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_cache_path = os.path.join(CACHE_DIR, "train_data.parquet")
    val_cache_path = os.path.join(CACHE_DIR, "val_data.parquet")
    test_cache_path = os.path.join(CACHE_DIR, "test_data.parquet")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            print("Loading datasets from cache...")
            df_train = pd.read_parquet(train_cache_path)
            df_val = pd.read_parquet(val_cache_path)
            df_test = pd.read_parquet(test_cache_path)
            return df_train, df_val, df_test

    print("Loading datasets from metadata CSVs...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Get all feature columns. We will ensure they are float64.
    # We get them unsorted here and rely on get_features_and_targets for the final sort,
    # but we need the list to cast types.
    feature_cols = get_feature_columns(sort_alphanumeric=False)

    # Enforce float64 precision for all feature columns
    for col in feature_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype(FLOAT_PRECISION)
        if col in df_val.columns:
            df_val[col] = df_val[col].astype(FLOAT_PRECISION)
        if col in df_test.columns:
            df_test[col] = df_test[col].astype(FLOAT_PRECISION)

    # Save processed dataframes to cache
    print(f"Saving processed datasets to cache at {CACHE_DIR}...")
    df_train.to_parquet(train_cache_path)
    df_val.to_parquet(val_cache_path)
    df_test.to_parquet(test_cache_path)

    return df_train, df_val, df_test


def get_features_and_targets(df, is_test=False):
    """
    Extracts the feature matrix X and target vector y (or ids) from the DataFrame.

    Explicitly applies alphanumeric sorting to feature columns to ensure
    consistent memory layout for the linear solver.

    Args:
        df (pd.DataFrame): The dataset containing features and targets/ids.
        is_test (bool): If True, returns (X, ids). If False, returns (X, y).

    Returns:
        tuple: (X, y) or (X, ids). X is a float64 numpy array.
    """
    # Retrieve feature columns and enforce alphanumeric sorting
    # e.g. margin_1, margin_10, margin_11 ...
    cols = get_feature_columns(sort_alphanumeric=False)
    sorted_cols = alphanumeric_sort(cols)

    # Extract features as a float64 numpy array
    X = df[sorted_cols].values.astype(FLOAT_PRECISION)

    if is_test:
        ids = df["id"].values
        return X, ids
    else:
        # For training/validation, return targets
        if "species" not in df.columns:
            raise ValueError(
                "DataFrame missing 'species' column required for training/validation."
            )
        y = df["species"].values
        return X, y
