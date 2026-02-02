import os
import ast
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    WORKING_DIR,
    LIST_COL,
    TARGET_COL,
    SEED,
)


def get_common_features(train_df, test_df, exclude_cols=None):
    """
    Identifies the intersection of columns between train and test dataframes
    to strictly prevent feature leakage.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        exclude_cols (list, optional): List of columns to exclude from the intersection
                                       (e.g., target variable). Defaults to None.

    Returns:
        list: Sorted list of common column names.
    """
    if exclude_cols is None:
        exclude_cols = []

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Find intersection
    common_cols = train_cols.intersection(test_cols)

    # Remove explicitly excluded columns
    common_cols = common_cols - set(exclude_cols)

    return sorted(list(common_cols))


def load_dataset(load_cached_data=True, debug_size=None):
    """
    Loads the train, validation, and test datasets. Implements caching using Parquet
    to preserve data types (specifically lists) and improve loading speed.

    Args:
        load_cached_data (bool): If True, attempts to load from Parquet cache.
                                 If False or cache missing, reloads from CSV and parses.
        debug_size (int, optional): If provided, samples the datasets to this size
                                    for debugging purposes.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_train_path = os.path.join(WORKING_DIR, "train_parsed.parquet")
    cache_val_path = os.path.join(WORKING_DIR, "val_parsed.parquet")
    cache_test_path = os.path.join(WORKING_DIR, "test_parsed.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    )

    if load_cached_data and cache_exists:
        print("Loading datasets from Parquet cache...")
        train_df = pd.read_parquet(cache_train_path)
        val_df = pd.read_parquet(cache_val_path)
        test_df = pd.read_parquet(cache_test_path)
    else:
        print("Loading datasets from raw CSVs and parsing structures...")

        # Load CSVs
        train_df = pd.read_csv(TRAIN_CSV)
        val_df = pd.read_csv(VAL_CSV)
        test_df = pd.read_csv(TEST_CSV)

        # Parse list columns
        # The metadata CSVs store lists as strings (e.g. "['a', 'b']").
        # We need to convert them back to actual lists.
        # We use ast.literal_eval for safe evaluation.

        def parse_list_col(df, col_name):
            if col_name in df.columns:
                # Fill NaNs with empty list string representation before parsing
                df[col_name] = df[col_name].fillna("[]")
                df[col_name] = df[col_name].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            return df

        train_df = parse_list_col(train_df, LIST_COL)
        val_df = parse_list_col(val_df, LIST_COL)
        test_df = parse_list_col(test_df, LIST_COL)

        # Save to cache
        print(f"Saving parsed datasets to cache at {WORKING_DIR}...")
        train_df.to_parquet(cache_train_path, index=False)
        val_df.to_parquet(cache_val_path, index=False)
        test_df.to_parquet(cache_test_path, index=False)

    # Handle Debugging (Subsampling)
    if debug_size is not None:
        print(f"Subsampling datasets to {debug_size} samples for debugging...")

        # Stratified sample for train/val if target exists
        if TARGET_COL in train_df.columns:
            # Ensure we don't sample more than available
            n_train = min(debug_size, len(train_df))
            train_df = (
                train_df.groupby(TARGET_COL, group_keys=False)
                .apply(
                    lambda x: x.sample(
                        int(np.rint(n_train * len(x) / len(train_df))),
                        random_state=SEED,
                    )
                )
                .reset_index(drop=True)
            )

        else:
            train_df = train_df.sample(
                n=min(debug_size, len(train_df)), random_state=SEED
            ).reset_index(drop=True)

        if TARGET_COL in val_df.columns:
            n_val = min(debug_size, len(val_df))
            val_df = (
                val_df.groupby(TARGET_COL, group_keys=False)
                .apply(
                    lambda x: x.sample(
                        int(np.rint(n_val * len(x) / len(val_df))), random_state=SEED
                    )
                )
                .reset_index(drop=True)
            )
        else:
            val_df = val_df.sample(
                n=min(debug_size, len(val_df)), random_state=SEED
            ).reset_index(drop=True)

        test_df = test_df.sample(
            n=min(debug_size, len(test_df)), random_state=SEED
        ).reset_index(drop=True)

    return train_df, val_df, test_df
