import os
import ast
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    HISTORY_COL,
    TARGET_COL,
)


def parse_list_column(df, col_name):
    """
    Parses a column containing string representations of lists into actual lists.
    """
    if col_name in df.columns:
        # Fill NaNs with empty list string representation and parse
        df[col_name] = (
            df[col_name]
            .fillna("[]")
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        )
    return df


def _load_single_split(
    csv_path: str, cache_filename: str, load_cached_data: bool
) -> pd.DataFrame:
    """
    Loads a single dataset split, handling caching logic.
    """
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing from source.")

    # 2. IF loading fails (file missing or corrupt) OR load_cached_data is False:
    print(f"Processing data from {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Process: Parse list columns
    df = parse_list_column(df, HISTORY_COL)

    # Process: Ensure target is integer if present (True/False -> 1/0)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Save the result to the cache directory
    os.makedirs(WORKING_DIR, exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
        print(f"Cached data saved to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


def load_data(load_cached_data: bool = True):
    """
    Loads the train, validation, and test datasets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed Parquet files.
                                 If False or cache missing, loads from metadata CSVs and re-processes.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = _load_single_split(
        TRAIN_METADATA_PATH, "train_parsed.parquet", load_cached_data
    )
    val_df = _load_single_split(
        VAL_METADATA_PATH, "val_parsed.parquet", load_cached_data
    )
    test_df = _load_single_split(
        TEST_METADATA_PATH, "test_parsed.parquet", load_cached_data
    )

    return train_df, val_df, test_df


def get_common_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame, exclude_cols: list = None
) -> list:
    """
    Identifies features present in both training and test datasets to prevent leakage.

    Args:
        df_train: Training DataFrame.
        df_test: Test DataFrame.
        exclude_cols: List of columns to explicitly exclude.

    Returns:
        list: List of column names common to both DataFrames (minus exclusions).
    """
    if exclude_cols is None:
        exclude_cols = []

    common_cols = [
        col
        for col in df_train.columns
        if col in df_test.columns and col not in exclude_cols
    ]
    return common_cols
