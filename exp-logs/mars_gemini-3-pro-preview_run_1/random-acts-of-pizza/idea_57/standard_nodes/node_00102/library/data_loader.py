import os
import ast
import pandas as pd
import numpy as np
from library import config
from library import utils


def parse_list_columns(df, columns):
    """
    Parses stringified list columns back into Python lists.

    Args:
        df (pd.DataFrame): The DataFrame containing columns to parse.
        columns (list): List of column names to parse.

    Returns:
        pd.DataFrame: The DataFrame with parsed columns.
    """
    for col in columns:
        if col in df.columns:
            # Use ast.literal_eval to safely parse string representation of lists
            # Handle cases where the value might already be a list or is NaN
            df[col] = df[col].apply(
                lambda x: (
                    ast.literal_eval(x)
                    if isinstance(x, str)
                    else (x if isinstance(x, list) else [])
                )
            )
    return df


def get_common_columns(
    train_df, test_df, target_col="requester_received_pizza", exclude_cols=None
):
    """
    Identifies common columns between train and test to prevent leakage.
    Excludes the target column and specific leakage columns.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Test DataFrame.
        target_col (str): The name of the target column to exclude from features.
        exclude_cols (list): List of additional columns to exclude (e.g., IDs).

    Returns:
        list: Sorted list of common feature column names.
    """
    if exclude_cols is None:
        exclude_cols = ["giver_username_if_known", "request_id", "source_file"]

    # Get intersection of columns
    common_cols = set(train_df.columns).intersection(set(test_df.columns))

    # Remove explicitly excluded columns
    common_cols = common_cols - set(exclude_cols)

    # Ensure target is not in the feature list
    if target_col in common_cols:
        common_cols.remove(target_col)

    return sorted(list(common_cols))


def load_dataset(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.
    Implements caching using Parquet to ensure fast loading and persistence.

    Args:
        load_cached_data (bool): If True, attempts to load from Parquet cache.
                                 If False or cache missing, loads from CSV and updates cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    cache_dir = config.CACHE_DIR
    # Ensure cache directory exists (redundant with config but safe)
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train.parquet")
    val_cache_path = os.path.join(cache_dir, "val.parquet")
    test_cache_path = os.path.join(cache_dir, "test.parquet")

    # Columns that contain list data stored as strings in CSV
    list_cols = ["requester_subreddits_at_request"]

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):

            print("Loading datasets from cache (Parquet)...")
            try:
                # Load parquet files
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Proceeding to load from source.")
        else:
            print("Cache files not found. Loading from source metadata CSVs...")
    else:
        print("Skipping cache load. Loading from source metadata CSVs...")

    # 2. Load from source CSVs
    print(f"Reading training data from {config.TRAIN_PATH}...")
    train_df = pd.read_csv(config.TRAIN_PATH)

    print(f"Reading validation data from {config.VAL_PATH}...")
    val_df = pd.read_csv(config.VAL_PATH)

    print(f"Reading test data from {config.TEST_PATH}...")
    test_df = pd.read_csv(config.TEST_PATH)

    # 3. Parse list columns
    print("Parsing stringified list columns...")
    train_df = parse_list_columns(train_df, list_cols)
    val_df = parse_list_columns(val_df, list_cols)
    test_df = parse_list_columns(test_df, list_cols)

    # 4. Save to cache
    print(f"Saving processed datasets to cache at {cache_dir}...")
    try:
        # Use pyarrow engine if available for better nested type support
        train_df.to_parquet(train_cache_path, index=False, engine="pyarrow")
        val_df.to_parquet(val_cache_path, index=False, engine="pyarrow")
        test_df.to_parquet(test_cache_path, index=False, engine="pyarrow")
    except Exception as e:
        print(f"Warning: Could not save to cache: {e}")
        # Fallback to default engine if pyarrow fails or is missing
        try:
            train_df.to_parquet(train_cache_path, index=False)
            val_df.to_parquet(val_cache_path, index=False)
            test_df.to_parquet(test_cache_path, index=False)
        except Exception as e2:
            print(f"Error: Failed to save cache with default engine as well: {e2}")

    return train_df, val_df, test_df
