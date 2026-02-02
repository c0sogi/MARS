import os
import pandas as pd
import numpy as np
from library.config import Config


def load_data(load_cached_data=True):
    """
    Loads train, validation, and test datasets.
    Implements caching using Parquet files to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from cached parquet files.
                                 If False or cache missing, reloads from metadata CSVs.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Check if cached files exist
    cache_exists = (
        os.path.exists(Config.CACHE_TRAIN_PROCESSED)
        and os.path.exists(Config.CACHE_VAL_PROCESSED)
        and os.path.exists(Config.CACHE_TEST_PROCESSED)
    )

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_df = pd.read_parquet(Config.CACHE_TRAIN_PROCESSED)
        val_df = pd.read_parquet(Config.CACHE_VAL_PROCESSED)
        test_df = pd.read_parquet(Config.CACHE_TEST_PROCESSED)
    else:
        print("Loading data from metadata CSVs...")
        # Load from metadata directory
        train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
        val_df = pd.read_csv(Config.VAL_DATA_PATH)
        test_df = pd.read_csv(Config.TEST_DATA_PATH)

        # Basic Cleaning
        # 1. Fill NaNs in text columns with empty strings to avoid errors in vectorization
        text_cols = [Config.TEXT_COL_TITLE, Config.TEXT_COL_BODY, "request_text"]
        for df in [train_df, val_df, test_df]:
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str)

        # 2. Convert Target to Integer (0/1) if present
        target_col = "requester_received_pizza"
        if target_col in train_df.columns:
            train_df[target_col] = train_df[target_col].astype(int)
        if target_col in val_df.columns:
            val_df[target_col] = val_df[target_col].astype(int)

        # Note: test_df might not have the target, or it might be dummy.
        # We don't enforce target conversion for test_df unless it exists.

        # Save to cache
        print("Saving processed data to cache...")
        train_df.to_parquet(Config.CACHE_TRAIN_PROCESSED, index=False)
        val_df.to_parquet(Config.CACHE_VAL_PROCESSED, index=False)
        test_df.to_parquet(Config.CACHE_TEST_PROCESSED, index=False)

    return train_df, val_df, test_df


def get_common_columns(train_df, test_df, exclude_cols=None):
    """
    Identifies the intersection of columns between train and test dataframes
    to prevent feature leakage and ensure model compatibility.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        exclude_cols (list, optional): List of columns to explicitly exclude
                                       (e.g., target, ID, leakage features).

    Returns:
        list: Sorted list of common column names.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Find intersection
    common_cols = set(train_df.columns).intersection(set(test_df.columns))

    # Remove excluded columns
    final_cols = [c for c in common_cols if c not in exclude_cols]

    return sorted(final_cols)


def save_submission(request_ids, probabilities):
    """
    Formats and saves the submission file.

    Args:
        request_ids (array-like): List of request_ids.
        probabilities (array-like): Predicted probabilities of success.
    """
    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": probabilities}
    )

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
