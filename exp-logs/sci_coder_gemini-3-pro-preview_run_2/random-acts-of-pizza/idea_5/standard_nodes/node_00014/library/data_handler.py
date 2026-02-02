import os
import json
import pandas as pd
import numpy as np
from library import config
from library.utils import setup_logger

logger = setup_logger("data_handler")


def _preprocess_dataframe(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Internal helper to clean and format the dataframe.
    - Fills missing text.
    - Creates 'combined_text'.
    - Selects relevant columns.
    """
    # 1. Handle Text Columns
    # Fill NaNs with empty string and ensure string type
    for col in config.TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            # Fallback if a column is missing (unlikely based on analysis)
            df[col] = ""

    # Combine text columns (Title + " " + Body)
    # We join them with a space separator.
    # This is crucial for the Siamese network input.
    df["combined_text"] = df[config.TEXT_COLS].apply(lambda x: " ".join(x), axis=1)

    # 2. Select Columns
    # We keep ID, combined_text, original text cols (optional, but good for debug), and numericals
    cols_to_keep = (
        [config.ID_COL, "combined_text"] + config.TEXT_COLS + config.NUMERICAL_COLS
    )

    if not is_test:
        if config.TARGET_COL in df.columns:
            cols_to_keep.append(config.TARGET_COL)
            # Ensure target is integer
            df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)
        else:
            logger.warning(
                f"Target column {config.TARGET_COL} not found in training/validation data."
            )

    # Filter columns, ignoring any that might be missing (though config should be correct)
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols]

    # 3. Handle Numerical NaNs
    # Although analysis showed no missing values, we fill with 0 for safety
    for col in config.NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def load_datasets(load_cached_data: bool = True):
    """
    Loads the train, validation, and test datasets.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from parquet cache.
                                 If False or cache miss, processes from scratch.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Define cache paths
    train_cache = config.TRAIN_FEATURES_PATH
    val_cache = config.VAL_FEATURES_PATH
    test_cache = config.TEST_FEATURES_PATH

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):

            logger.info("Loading datasets from cache...")
            try:
                df_train = pd.read_parquet(train_cache)
                df_val = pd.read_parquet(val_cache)
                df_test = pd.read_parquet(test_cache)
                logger.info(
                    f"Loaded train: {df_train.shape}, val: {df_val.shape}, test: {df_test.shape}"
                )
                return df_train, df_val, df_test
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reprocessing from scratch.")
        else:
            logger.info("Cache files not found. Processing from scratch.")
    else:
        logger.info("load_cached_data is False. Processing from scratch.")

    # 2. Load Raw Data
    logger.info("Loading raw JSON data...")
    with open(config.TRAIN_JSON_PATH, "r") as f:
        raw_train_data = json.load(f)
    with open(config.TEST_JSON_PATH, "r") as f:
        raw_test_data = json.load(f)

    # Convert to DataFrames
    df_raw_train = pd.DataFrame(raw_train_data)
    df_raw_test = pd.DataFrame(raw_test_data)

    # 3. Load Metadata
    logger.info("Loading metadata...")
    meta_train = pd.read_csv(config.TRAIN_META_PATH)
    meta_val = pd.read_csv(config.VAL_META_PATH)
    meta_test = pd.read_csv(config.TEST_META_PATH)

    # 4. Merge Data
    # We merge metadata with raw data on request_id.
    # The raw training file contains both train and val samples.
    logger.info("Merging and splitting data...")

    # Train Split
    df_train = meta_train.merge(df_raw_train, on=config.ID_COL, how="left")
    # Resolve target column conflicts if they exist (e.g., from metadata vs raw)
    if "requester_received_pizza_x" in df_train.columns:
        df_train[config.TARGET_COL] = df_train["requester_received_pizza_x"]
        df_train.drop(
            columns=["requester_received_pizza_x", "requester_received_pizza_y"],
            inplace=True,
            errors="ignore",
        )

    # Val Split
    df_val = meta_val.merge(df_raw_train, on=config.ID_COL, how="left")
    if "requester_received_pizza_x" in df_val.columns:
        df_val[config.TARGET_COL] = df_val["requester_received_pizza_x"]
        df_val.drop(
            columns=["requester_received_pizza_x", "requester_received_pizza_y"],
            inplace=True,
            errors="ignore",
        )

    # Test Split
    df_test = meta_test.merge(df_raw_test, on=config.ID_COL, how="left")

    # 5. Preprocess
    logger.info("Preprocessing datasets...")
    df_train = _preprocess_dataframe(df_train, is_test=False)
    df_val = _preprocess_dataframe(df_val, is_test=False)
    df_test = _preprocess_dataframe(df_test, is_test=True)

    # 6. Save to Cache
    logger.info("Saving processed datasets to cache...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(train_cache), exist_ok=True)

    df_train.to_parquet(train_cache, index=False)
    df_val.to_parquet(val_cache, index=False)
    df_test.to_parquet(test_cache, index=False)

    logger.info(
        f"Processing complete. Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
    )

    return df_train, df_val, df_test
