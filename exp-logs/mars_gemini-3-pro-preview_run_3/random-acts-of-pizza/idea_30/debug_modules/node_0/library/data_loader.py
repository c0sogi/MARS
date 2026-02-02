import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import log


def load_data(load_cached_data: bool = True, debug_size: int = None):
    """
    Loads the train, validation, and test data from metadata or cache.
    Performs leakage prevention by removing retrieval-time columns.
    Standardizes text columns to use the edit-aware version.

    Args:
        load_cached_data (bool): If True, attempts to load from local parquet cache.
        debug_size (int, optional): If set, truncates data to this number of rows.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "cleaned_train.parquet")
    val_cache = os.path.join(cache_dir, "cleaned_val.parquet")
    test_cache = os.path.join(cache_dir, "cleaned_test.parquet")

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        log("Loading cleaned data from cache...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        log("Loading data from metadata splits...")
        # Load from metadata
        train_df = pd.read_parquet(Config.TRAIN_DATA_PATH)
        val_df = pd.read_parquet(Config.VAL_DATA_PATH)
        test_df = pd.read_parquet(Config.TEST_DATA_PATH)

        # Preprocess (Cleaning & Leakage Removal)
        log("Preprocessing data: Removing leakage and standardizing text...")
        train_df = _preprocess_dataframe(train_df)
        val_df = _preprocess_dataframe(val_df)
        test_df = _preprocess_dataframe(test_df)

        # Save to cache
        log(f"Saving cleaned data to cache at {cache_dir}...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    # Handle Debugging
    if debug_size is not None:
        log(f"Debug Mode: Subsampling data to {debug_size} rows.")
        train_df = train_df.iloc[:debug_size].copy()
        val_df = val_df.iloc[:debug_size].copy()
        test_df = test_df.iloc[:debug_size].copy()

    return train_df, val_df, test_df


def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper to clean a single dataframe.
    1. Standardizes 'request_text_edit_aware' -> 'request_text'.
    2. Drops columns ending in '_at_retrieval' to prevent leakage.
    """
    df = df.copy()

    # 1. Text Standardization
    # We prefer the edit-aware text to avoid leakage (e.g. "Edit: Thanks for the pizza")
    if "request_text_edit_aware" in df.columns:
        # If the raw text column exists, drop it to avoid confusion/duplication
        if "request_text" in df.columns:
            df.drop(columns=["request_text"], inplace=True)
        # Rename edit_aware to standard 'request_text'
        df.rename(columns={"request_text_edit_aware": "request_text"}, inplace=True)

    # 2. Leakage Prevention
    # Drop any column that represents future information collected at retrieval time
    retrieval_cols = [c for c in df.columns if c.endswith("_at_retrieval")]
    if retrieval_cols:
        df.drop(columns=retrieval_cols, inplace=True)

    return df


def get_data_splits(load_cached_data: bool = True, debug_size: int = None):
    """
    High-level function to get X and y splits for Train, Val, and Test.

    Args:
        load_cached_data (bool): Whether to use cached cleaned data.
        debug_size (int, optional): Number of rows for debugging.

    Returns:
        tuple: (train_df, y_train, val_df, y_val, test_df, test_ids)
            - train_df (pd.DataFrame): Training features (target removed).
            - y_train (pd.Series): Training target.
            - val_df (pd.DataFrame): Validation features (target removed).
            - y_val (pd.Series): Validation target.
            - test_df (pd.DataFrame): Test features.
            - test_ids (pd.Series): Request IDs for submission.
    """
    train_df, val_df, test_df = load_data(
        load_cached_data=load_cached_data, debug_size=debug_size
    )

    target_col = "requester_received_pizza"
    id_col = "request_id"

    # Prepare Train
    if target_col in train_df.columns:
        y_train = train_df[target_col].astype(int)
        train_df = train_df.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' missing from training data.")

    # Prepare Val
    if target_col in val_df.columns:
        y_val = val_df[target_col].astype(int)
        val_df = val_df.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' missing from validation data.")

    # Prepare Test IDs
    if id_col in test_df.columns:
        test_ids = test_df[id_col]
    else:
        raise ValueError(f"ID column '{id_col}' missing from test data.")

    log(
        f"Data Loaded. Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )

    return train_df, y_train, val_df, y_val, test_df, test_ids
