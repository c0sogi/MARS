import os
import ast
import pandas as pd
import numpy as np
from library.config import TRAIN_PATH, VAL_PATH, TEST_PATH, WORKING_DIR, RANDOM_STATE


def parse_list_column(df, col_name):
    """
    Parses a column containing string representations of lists into actual Python lists.
    Used for columns like 'requester_subreddits_at_request' which are stringified in CSVs.
    """
    if col_name in df.columns and not df.empty:
        # Check the first non-null value to see if it looks like a stringified list
        sample = (
            df[col_name].dropna().iloc[0] if not df[col_name].dropna().empty else ""
        )
        if (
            isinstance(sample, str)
            and sample.strip().startswith("[")
            and sample.strip().endswith("]")
        ):
            try:
                # Apply literal_eval safely
                df[col_name] = df[col_name].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            except (ValueError, SyntaxError):
                # If parsing fails, leave column as is
                pass
    return df


def get_common_columns(
    df_train, df_test, target_col="requester_received_pizza", exclude_cols=None
):
    """
    Identifies the intersection of columns between training and test sets to ensure
    consistent feature input and prevent leakage.

    Args:
        df_train (pd.DataFrame): Training dataframe.
        df_test (pd.DataFrame): Test dataframe.
        target_col (str): Name of the target variable to exclude from feature list.
        exclude_cols (list, optional): List of additional columns to exclude (e.g., identifiers).

    Returns:
        list: Sorted list of column names present in both dataframes.
    """
    if exclude_cols is None:
        exclude_cols = ["request_id", "source_file", "giver_username_if_known"]

    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    # Find intersection
    common_cols = train_cols.intersection(test_cols)

    # Remove target column if present (it shouldn't be in inputs)
    if target_col in common_cols:
        common_cols.remove(target_col)

    # Remove manually excluded columns
    for col in exclude_cols:
        if col in common_cols:
            common_cols.remove(col)

    return sorted(list(common_cols))


def load_dataset(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.

    Logic:
    1. Checks if Parquet cache exists in WORKING_DIR.
    2. If exists and load_cached_data is True, loads from Parquet.
    3. Otherwise, loads from original Metadata CSVs.
    4. Parses stringified list columns.
    5. Saves to Parquet cache for future use.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    train_cache = os.path.join(WORKING_DIR, "train_loader.parquet")
    val_cache = os.path.join(WORKING_DIR, "val_loader.parquet")
    test_cache = os.path.join(WORKING_DIR, "test_loader.parquet")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            try:
                df_train = pd.read_parquet(train_cache)
                df_val = pd.read_parquet(val_cache)
                df_test = pd.read_parquet(test_cache)
                return df_train, df_val, df_test
            except Exception:
                # Fallback to loading from source if cache is corrupt
                pass

    # Load from Metadata CSVs
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Parse columns that contain lists (stored as strings in CSV)
    # 'requester_subreddits_at_request' is critical for community profiling
    list_columns = ["requester_subreddits_at_request"]

    for col in list_columns:
        df_train = parse_list_column(df_train, col)
        df_val = parse_list_column(df_val, col)
        df_test = parse_list_column(df_test, col)

    # Save to cache
    try:
        df_train.to_parquet(train_cache)
        df_val.to_parquet(val_cache)
        df_test.to_parquet(test_cache)
    except Exception:
        # Proceed without caching if write fails
        pass

    return df_train, df_val, df_test
