import os
import pandas as pd
import numpy as np
import ast
from library import config


def get_common_columns(train_df, test_df, target_col=None):
    """
    Identifies the intersection of columns between train and test dataframes
    to prevent feature leakage.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        target_col (str, optional): The target column name to exclude from the feature list.

    Returns:
        list: A sorted list of column names present in both dataframes.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    common_cols = list(train_cols.intersection(test_cols))

    if target_col and target_col in common_cols:
        common_cols.remove(target_col)

    return sorted(common_cols)


def parse_list_column(df, column_name):
    """
    Parses a stringified list column into actual Python lists.
    Handles cases where the column might already be parsed or contains NaNs.
    """
    if column_name not in df.columns:
        return df

    def safe_eval(x):
        if isinstance(x, list):
            return x
        if pd.isna(x) or x == "" or x == "[]":
            return []
        try:
            # The data seems to be Python-style lists in string format
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    df[column_name] = df[column_name].apply(safe_eval)
    return df


def load_datasets(load_cached_data=True):
    """
    Loads train, validation, and test datasets.
    Implements caching using Parquet files to optimize runtime.

    Args:
        load_cached_data (bool): If True, attempts to load from cached Parquet files.
                                 If False or cache missing, loads from original CSVs.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train.parquet")
    val_cache = os.path.join(cache_dir, "val.parquet")
    test_cache = os.path.join(cache_dir, "test.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        print("Loading datasets from cache...")
        try:
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    print("Loading datasets from source CSVs...")

    # Load from metadata CSVs
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    # Preprocessing: Parse 'requester_subreddits_at_request' from string to list
    # This is necessary because CSVs store lists as strings
    list_col = "requester_subreddits_at_request"
    print(f"Parsing list column: {list_col}")
    train_df = parse_list_column(train_df, list_col)
    val_df = parse_list_column(val_df, list_col)
    test_df = parse_list_column(test_df, list_col)

    # Ensure target column is boolean in train/val
    target_col = "requester_received_pizza"
    if target_col in train_df.columns:
        train_df[target_col] = train_df[target_col].astype(bool)
    if target_col in val_df.columns:
        val_df[target_col] = val_df[target_col].astype(bool)

    # Save to cache
    print("Saving datasets to cache...")
    try:
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return train_df, val_df, test_df
