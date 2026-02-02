import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("data_loader")


def load_and_process_data(debug: bool = False, load_cached_data: bool = True):
    """
    Loads and processes the dataset.

    1. Tries to load from Parquet cache if available and enabled.
    2. If not, loads raw JSON and Metadata CSVs.
    3. Merges metadata with raw data to create Train/Val/Test splits.
    4. Selects relevant columns (Text + Numerical + Target).
    5. Saves processed dataframes to Parquet cache.
    6. Applies debug sampling if requested.

    Args:
        debug (bool): If True, subsamples the data for debugging.
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        tuple: (df_train, df_val, df_test)
    """

    # Define cache paths
    train_cache = Config.TRAIN_FEATURES_PATH
    val_cache = Config.VAL_FEATURES_PATH
    test_cache = Config.TEST_FEATURES_PATH

    # ---------------------------------------------------------
    # 1. Try Loading from Cache
    # ---------------------------------------------------------
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            logger.info("Loading data from cache...")
            df_train = pd.read_parquet(train_cache)
            df_val = pd.read_parquet(val_cache)
            df_test = pd.read_parquet(test_cache)

            if debug:
                logger.info(
                    f"Debug mode: Subsampling to {Config.DEBUG_SAMPLE_SIZE} samples."
                )
                df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
                df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
                df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

            return df_train, df_val, df_test
        else:
            logger.info("Cache files not found. Processing from scratch...")
    else:
        logger.info("Ignoring cache. Processing from scratch...")

    # ---------------------------------------------------------
    # 2. Load Raw Data and Metadata
    # ---------------------------------------------------------
    logger.info("Loading raw JSON data...")
    with open(Config.TRAIN_DATA_PATH, "r") as f:
        raw_train_list = json.load(f)
    with open(Config.TEST_DATA_PATH, "r") as f:
        raw_test_list = json.load(f)

    df_raw_train = pd.DataFrame(raw_train_list)
    df_raw_test = pd.DataFrame(raw_test_list)

    logger.info("Loading metadata splits...")
    meta_train = pd.read_csv(Config.TRAIN_META_PATH)
    meta_val = pd.read_csv(Config.VAL_META_PATH)
    meta_test = pd.read_csv(Config.TEST_META_PATH)

    # ---------------------------------------------------------
    # 3. Merge and Process Splits
    # ---------------------------------------------------------
    # Define columns to keep
    # ID + Target + Text + Numerical
    cols_to_keep = [Config.ID_COL] + Config.TEXT_COLS + Config.NUMERICAL_COLS

    # Helper to process a split
    def process_split(meta_df, raw_df, is_test=False):
        # Merge metadata with raw data on request_id
        # We use inner join to keep only rows defined in the metadata split
        merged = meta_df.merge(
            raw_df, on=Config.ID_COL, how="inner", suffixes=("", "_raw")
        )

        # Select columns
        current_cols = cols_to_keep.copy()

        # Add target if not test
        if not is_test:
            # Ensure target is in the list
            if Config.TARGET_COL not in current_cols:
                current_cols.append(Config.TARGET_COL)

        # Filter columns
        # Note: Some columns might be in metadata or raw, merge handles this.
        # We prioritize the columns from raw_df usually, but target comes from metadata/raw intersection.
        # Check if columns exist
        available_cols = [c for c in current_cols if c in merged.columns]
        missing_cols = set(current_cols) - set(available_cols)
        if missing_cols:
            logger.warning(f"Missing columns in split: {missing_cols}")

        df_out = merged[available_cols].copy()

        # Data Cleaning / Type Casting

        # 1. Text: Fill NaNs with empty string
        for col in Config.TEXT_COLS:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna("").astype(str)

        # 2. Numerical: Fill NaNs with 0 (safe default for counts/timestamps in this context)
        for col in Config.NUMERICAL_COLS:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(0).astype(float)

        # 3. Target: Convert to int if present
        if not is_test and Config.TARGET_COL in df_out.columns:
            df_out[Config.TARGET_COL] = df_out[Config.TARGET_COL].astype(int)

        return df_out

    logger.info("Processing Train split...")
    df_train = process_split(meta_train, df_raw_train, is_test=False)

    logger.info("Processing Validation split...")
    df_val = process_split(meta_val, df_raw_train, is_test=False)

    logger.info("Processing Test split...")
    df_test = process_split(meta_test, df_raw_test, is_test=True)

    # ---------------------------------------------------------
    # 4. Save to Cache
    # ---------------------------------------------------------
    logger.info(f"Saving processed data to {Config.WORKING_DIR}...")
    df_train.to_parquet(train_cache, index=False)
    df_val.to_parquet(val_cache, index=False)
    df_test.to_parquet(test_cache, index=False)

    # ---------------------------------------------------------
    # 5. Debug Handling
    # ---------------------------------------------------------
    if debug:
        logger.info(f"Debug mode: Subsampling to {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    logger.info(
        f"Data processing complete. Train shape: {df_train.shape}, Val shape: {df_val.shape}, Test shape: {df_test.shape}"
    )
    return df_train, df_val, df_test
