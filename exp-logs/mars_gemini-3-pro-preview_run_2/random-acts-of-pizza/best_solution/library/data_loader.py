import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, load_data_splits, save_parquet, load_parquet

logger = setup_logger("data_loader")


def process_dataframe(df, is_test=False):
    """
    Cleans and selects relevant columns for the MF-ADBE architecture.

    Args:
        df (pd.DataFrame): The raw dataframe.
        is_test (bool): Whether this is the test set (excludes target).

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # 1. Text Handling
    # Fill NaNs with empty strings to ensure tokenizers don't crash
    text_cols = [Config.TEXT_COL_TITLE, Config.TEXT_COL_BODY]
    for col in text_cols:
        if col in df.columns:
            # Ensure string type and handle NaNs
            df[col] = df[col].fillna("").astype(str)
        else:
            # Fallback if column missing (unlikely)
            df[col] = ""

    # 2. Numerical Features
    # Ensure they exist and are numeric, filling missing with 0
    for col in Config.NUMERICAL_FEATURES:
        if col not in df.columns:
            logger.warning(
                f"Numerical feature {col} missing in dataframe. Filling with 0."
            )
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 3. Select Columns
    # We keep request_id for alignment
    cols_to_keep = ["request_id"] + text_cols + Config.NUMERICAL_FEATURES

    if not is_test:
        if "requester_received_pizza" in df.columns:
            cols_to_keep.append("requester_received_pizza")
        else:
            logger.warning(
                "Target column 'requester_received_pizza' missing in training/val data."
            )

    # Filter columns
    df_processed = df[cols_to_keep].copy()

    return df_processed


def load_and_process_data(load_cached_data=True):
    """
    Loads raw data, processes it (filling NaNs, selecting columns),
    and returns train, val, and test DataFrames.
    Implements caching via Parquet files.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Define cache paths from Config
    train_cache = Config.CACHE_TRAIN_FEATURES
    val_cache = Config.CACHE_VAL_FEATURES
    test_cache = Config.CACHE_TEST_FEATURES

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):

            logger.info("Loading processed data from cache...")
            try:
                df_train = load_parquet(train_cache)
                df_val = load_parquet(val_cache)
                df_test = load_parquet(test_cache)
                return df_train, df_val, df_test
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")
        else:
            logger.info("Cache not found. Processing from scratch...")
    else:
        logger.info("Ignoring cache. Processing from scratch...")

    # Load raw splits using utility function
    logger.info("Loading raw data splits...")
    df_train_raw, df_val_raw, df_test_raw = load_data_splits()

    # Process each split
    logger.info("Processing training data...")
    df_train = process_dataframe(df_train_raw, is_test=False)

    logger.info("Processing validation data...")
    df_val = process_dataframe(df_val_raw, is_test=False)

    logger.info("Processing test data...")
    df_test = process_dataframe(df_test_raw, is_test=True)

    # Save to cache
    logger.info(f"Saving processed data to {Config.WORKING_DIR}...")
    save_parquet(df_train, train_cache)
    save_parquet(df_val, val_cache)
    save_parquet(df_test, test_cache)

    return df_train, df_val, df_test
