import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, save_to_cache, load_from_cache

# Initialize Logger
logger = setup_logger(
    "data_loader", os.path.join(Config.WORKING_DIR, "data_loader.log")
)


def load_dataset(load_cached_data: bool = True):
    """
    Loads the dataset, merging raw JSON data with metadata splits.
    Implements caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_merged.parquet")
    val_cache_path = os.path.join(cache_dir, "val_merged.parquet")
    test_cache_path = os.path.join(cache_dir, "test_merged.parquet")

    # 1. Try Loading from Cache
    if load_cached_data:
        logger.info("Attempting to load datasets from cache...")
        df_train = load_from_cache(train_cache_path)
        df_val = load_from_cache(val_cache_path)
        df_test = load_from_cache(test_cache_path)

        if df_train is not None and df_val is not None and df_test is not None:
            logger.info("Successfully loaded datasets from cache.")
            return df_train, df_val, df_test
        else:
            logger.info("Cache miss or incomplete cache. Loading from raw sources.")

    # 2. Load Metadata
    logger.info("Loading metadata CSVs...")
    df_meta_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_meta_val = pd.read_csv(Config.VAL_META_PATH)
    df_meta_test = pd.read_csv(Config.TEST_META_PATH)

    # 3. Load Raw Data
    logger.info("Loading raw JSON files...")
    with open(Config.TRAIN_DATA_PATH, "r") as f:
        raw_train_data = json.load(f)

    with open(Config.TEST_DATA_PATH, "r") as f:
        raw_test_data = json.load(f)

    # Convert raw data to DataFrames
    df_raw_train = pd.DataFrame(raw_train_data)
    df_raw_test = pd.DataFrame(raw_test_data)

    # 4. Merge Metadata with Raw Data
    # We use 'request_id' as the key.
    # Note: The raw train file contains both train and val samples (and potentially others),
    # so merging with metadata filters it down to the specific split.
    logger.info("Merging metadata with raw data...")

    # Merge Train
    # Metadata has 'requester_received_pizza', raw might have it too.
    # We prioritize metadata for the split definition, but raw has features.
    # We drop 'requester_received_pizza' from raw to avoid suffixes if it exists.
    cols_to_drop_raw = (
        ["requester_received_pizza"]
        if "requester_received_pizza" in df_raw_train.columns
        else []
    )
    df_train = df_meta_train.merge(
        df_raw_train.drop(columns=cols_to_drop_raw, errors="ignore"),
        on="request_id",
        how="left",
    )

    # Merge Val
    df_val = df_meta_val.merge(
        df_raw_train.drop(columns=cols_to_drop_raw, errors="ignore"),
        on="request_id",
        how="left",
    )

    # Merge Test
    # Test metadata does not have the target. Raw test does not have the target.
    df_test = df_meta_test.merge(df_raw_test, on="request_id", how="left")

    # 5. Select and Clean Columns
    logger.info("Selecting and cleaning features...")

    # Define columns to keep
    # Base columns
    keep_cols = ["request_id"]

    # Text columns
    keep_cols.append(Config.TEXT_COL_TITLE)
    keep_cols.append(Config.TEXT_COL_BODY)

    # Numerical columns
    keep_cols.extend(Config.NUMERICAL_FEATURES)

    # Target (only for train/val)
    target_col = "requester_received_pizza"

    def process_df(df, is_test=False):
        # Select columns
        current_keep = keep_cols.copy()
        if not is_test:
            current_keep.append(target_col)

        # Ensure columns exist (handle potential missing columns in raw data gracefully)
        available_cols = [c for c in current_keep if c in df.columns]
        df_subset = df[available_cols].copy()

        # Fill missing text with empty string
        if Config.TEXT_COL_TITLE in df_subset.columns:
            df_subset[Config.TEXT_COL_TITLE] = (
                df_subset[Config.TEXT_COL_TITLE].fillna("").astype(str)
            )
        if Config.TEXT_COL_BODY in df_subset.columns:
            df_subset[Config.TEXT_COL_BODY] = (
                df_subset[Config.TEXT_COL_BODY].fillna("").astype(str)
            )

        # Fill missing numericals with 0 (though analysis showed none, this is safe)
        for col in Config.NUMERICAL_FEATURES:
            if col in df_subset.columns:
                df_subset[col] = df_subset[col].fillna(0).astype(float)

        # Cast target to int if present
        if not is_test and target_col in df_subset.columns:
            df_subset[target_col] = df_subset[target_col].astype(int)

        return df_subset

    df_train = process_df(df_train, is_test=False)
    df_val = process_df(df_val, is_test=False)
    df_test = process_df(df_test, is_test=True)

    # 6. Debug Sampling
    if Config.DEBUG_SAMPLE_SIZE is not None:
        logger.info(f"Debug mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # 7. Save to Cache
    logger.info("Saving processed datasets to cache...")
    save_to_cache(df_train, train_cache_path)
    save_to_cache(df_val, val_cache_path)
    save_to_cache(df_test, test_cache_path)

    logger.info(
        f"Data loading complete. Train shape: {df_train.shape}, Val shape: {df_val.shape}, Test shape: {df_test.shape}"
    )

    return df_train, df_val, df_test
