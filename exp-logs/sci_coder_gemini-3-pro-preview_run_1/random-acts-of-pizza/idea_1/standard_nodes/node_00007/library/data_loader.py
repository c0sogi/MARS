import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    WORKING_DIR,
    NUMERIC_COLS,
    TEXT_COLS,
    ID_COL,
    TARGET_COL,
    RANDOM_STATE,
)
from library.utils import set_seed


def get_feature_intersection(train_df, test_df, exclude_cols=None):
    """
    Identifies the subset of columns present in both training and test sets.
    This effectively filters out 'at_retrieval' leakage features that exist
    only in the training data.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        exclude_cols (list, optional): List of columns to explicitly exclude (e.g., target).

    Returns:
        list: List of column names present in both dataframes.
    """
    if exclude_cols is None:
        exclude_cols = []

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Intersection of columns
    common_cols = list(train_cols.intersection(test_cols))

    # Remove excluded columns
    final_cols = [col for col in common_cols if col not in exclude_cols]

    return sorted(final_cols)


def load_data(load_cached_data=True, debug_sample_size=None):
    """
    Loads, processes, and returns the train, validation, and test datasets.
    Implements caching using Parquet files.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
        debug_sample_size (int, optional): If set, limits the dataset size for debugging.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    set_seed(RANDOM_STATE)

    # Define cache paths
    cache_train_path = os.path.join(WORKING_DIR, "train_processed.parquet")
    cache_val_path = os.path.join(WORKING_DIR, "val_processed.parquet")
    cache_test_path = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Check if cache exists and is requested
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):
            print("Loading data from cache...")
            train_df = pd.read_parquet(cache_train_path)
            val_df = pd.read_parquet(cache_val_path)
            test_df = pd.read_parquet(cache_test_path)
            return train_df, val_df, test_df
        else:
            print("Cache not found or incomplete. Processing from scratch...")
    else:
        print("Ignoring cache. Processing from scratch...")

    # Load raw data
    print("Loading raw metadata CSVs...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Debugging: Subsample if requested
    if debug_sample_size is not None:
        print(f"Subsampling data to {debug_sample_size} samples for debugging...")
        train_df = train_df.sample(
            n=min(len(train_df), debug_sample_size), random_state=RANDOM_STATE
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_sample_size), random_state=RANDOM_STATE
        ).reset_index(drop=True)
        # We usually don't subsample test for submission, but for full pipeline debugging we might.
        # However, to ensure valid submission generation code, we generally keep test intact or subsample it too.
        # Here we subsample test as well to speed up the whole pipeline if debugging.
        test_df = test_df.sample(
            n=min(len(test_df), debug_sample_size), random_state=RANDOM_STATE
        ).reset_index(drop=True)

    # 1. Text Processing
    # Combine title and text into a single column
    print("Processing text features...")
    for df in [train_df, val_df, test_df]:
        # Fill NaNs with empty string to allow concatenation
        df[TEXT_COLS[0]] = df[TEXT_COLS[0]].fillna("")
        df[TEXT_COLS[1]] = df[TEXT_COLS[1]].fillna("")

        # Create combined text column
        df["combined_text"] = df[TEXT_COLS[0]] + " " + df[TEXT_COLS[1]]

    # Feature Engineering
    print("Engineering features...")
    for df in [train_df, val_df, test_df]:
        # Upvotes ratio: (Diff / Sum)
        # Avoid div by zero
        sum_votes = df["requester_upvotes_plus_downvotes_at_request"].replace(0, 1)
        diff_votes = df["requester_upvotes_minus_downvotes_at_request"]
        df["requester_upvotes_ratio_at_request"] = diff_votes / sum_votes

        # RAOP comment ratio
        raop_comments = df["requester_number_of_comments_in_raop_at_request"]
        total_comments = df["requester_number_of_comments_at_request"].replace(0, 1)
        df["requester_raop_comment_ratio_at_request"] = raop_comments / total_comments

        # Text length feature
        df["request_text_length"] = df["combined_text"].apply(len)

    # 2. Feature Selection
    # Identify intersection of columns to avoid leakage (features not in test)
    # We exclude ID and Target from this check as we handle them explicitly
    intersection_cols = get_feature_intersection(
        train_df, test_df, exclude_cols=[ID_COL, TARGET_COL]
    )

    # Filter intersection cols to only include those we intend to use (NUMERIC_COLS)
    # This ensures we don't accidentally include metadata we don't want (like 'source_file')
    # effectively validating that our config NUMERIC_COLS are present in both.
    valid_numeric_cols = [c for c in intersection_cols if c in NUMERIC_COLS]

    print(
        f"Selected {len(valid_numeric_cols)} numeric features based on train/test intersection."
    )

    # Define final output columns
    # Train/Val have Target, Test does not.
    # We keep ID_COL for submission generation.

    cols_to_keep_train = [ID_COL, TARGET_COL, "combined_text"] + valid_numeric_cols
    cols_to_keep_test = [ID_COL, "combined_text"] + valid_numeric_cols

    train_df = train_df[cols_to_keep_train].copy()
    val_df = val_df[cols_to_keep_train].copy()
    test_df = test_df[cols_to_keep_test].copy()

    # 3. Caching
    print("Saving processed data to cache...")
    os.makedirs(WORKING_DIR, exist_ok=True)
    train_df.to_parquet(cache_train_path, index=False)
    val_df.to_parquet(cache_val_path, index=False)
    test_df.to_parquet(cache_test_path, index=False)

    return train_df, val_df, test_df
