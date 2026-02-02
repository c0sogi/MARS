import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import Timer


def load_raw_datasets():
    """
    Loads the raw train, validation, and test datasets from the metadata directory.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    with Timer("Load Raw Datasets"):
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        val_df = pd.read_parquet(Config.VAL_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)

    return train_df, val_df, test_df


def clean_data(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Cleans the dataframe by removing leakage columns and handling missing text.

    Args:
        df (pd.DataFrame): The dataframe to clean.
        is_test (bool): Whether this is the test set (target column handling).

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    # 1. Leakage Prevention: Drop columns suffixed with '_at_retrieval'
    # These features represent future information relative to the request time.
    retrieval_cols = [c for c in df.columns if c.endswith("_at_retrieval")]
    if retrieval_cols:
        df = df.drop(columns=retrieval_cols)

    # 2. Text Handling: Filter/Clean rows based on 'request_text_edit_aware'
    # Ensure the column exists and handle NaNs
    if "request_text_edit_aware" in df.columns:
        # Fill NaNs with empty string to avoid errors in downstream text processing
        df["request_text_edit_aware"] = df["request_text_edit_aware"].fillna("")

    # 3. Ensure 'request_title' is also clean
    if "request_title" in df.columns:
        df["request_title"] = df["request_title"].fillna("")

    return df


def load_and_clean_data(load_cached_data: bool = True):
    """
    Main data loading function with caching mechanism.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.
                                 If False or cache miss, re-processes and saves.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_train_path = os.path.join(Config.WORKING_DIR, "cleaned_train.parquet")
    cache_val_path = os.path.join(Config.WORKING_DIR, "cleaned_val.parquet")
    cache_test_path = os.path.join(Config.WORKING_DIR, "cleaned_test.parquet")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):

            print(f"Loading cached data from {Config.WORKING_DIR}...")
            with Timer("Load Cached Data"):
                train_df = pd.read_parquet(cache_train_path)
                val_df = pd.read_parquet(cache_val_path)
                test_df = pd.read_parquet(cache_test_path)
            return train_df, val_df, test_df
        else:
            print("Cache not found. Processing from scratch...")
    else:
        print("Ignoring cache. Processing from scratch...")

    # Load raw data
    train_df, val_df, test_df = load_raw_datasets()

    # Apply cleaning
    with Timer("Clean Data"):
        train_df = clean_data(train_df, is_test=False)
        val_df = clean_data(val_df, is_test=False)
        test_df = clean_data(test_df, is_test=True)

    # Save to cache
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    with Timer("Save Cache"):
        train_df.to_parquet(cache_train_path, index=False)
        val_df.to_parquet(cache_val_path, index=False)
        test_df.to_parquet(cache_test_path, index=False)

    return train_df, val_df, test_df
