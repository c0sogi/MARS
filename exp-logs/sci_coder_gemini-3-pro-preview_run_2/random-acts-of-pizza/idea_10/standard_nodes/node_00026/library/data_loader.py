import os
import json
import pandas as pd
from library.config import Config
from library.utils import setup_logger

# Initialize Logger
logger = setup_logger("data_loader")


def load_dataset(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.

    If load_cached_data is True and the parquet files exist in the working directory,
    it loads from there. Otherwise, it loads from the raw JSON and metadata CSVs,
    merges them, saves to parquet, and returns the dataframes.

    Args:
        load_cached_data (bool): Whether to attempt loading from cached parquet files.

    Returns:
        tuple: (df_train, df_val, df_test)
    """

    # Define cache paths from Config
    train_cache = Config.TRAIN_FEATURES_PATH
    val_cache = Config.VAL_FEATURES_PATH
    test_cache = Config.TEST_FEATURES_PATH

    # 1. Try loading from cache
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

                # Handle Debug mode even when loading from cache
                if Config.DEBUG:
                    logger.info(
                        f"DEBUG mode enabled. Sampling {Config.DEBUG_SAMPLES} rows."
                    )
                    df_train = df_train.head(Config.DEBUG_SAMPLES)
                    df_val = df_val.head(Config.DEBUG_SAMPLES)
                    df_test = df_test.head(Config.DEBUG_SAMPLES)

                return df_train, df_val, df_test
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reloading from source.")
        else:
            logger.info("Cache files not found. Loading from source.")

    # 2. Load from Source
    logger.info("Loading raw JSON data...")

    # Load raw JSON files
    with open(Config.TRAIN_JSON, "r") as f:
        raw_train_data = json.load(f)
    df_raw_train = pd.DataFrame(raw_train_data)

    with open(Config.TEST_JSON, "r") as f:
        raw_test_data = json.load(f)
    df_raw_test = pd.DataFrame(raw_test_data)

    logger.info("Loading metadata splits...")
    # Load metadata CSVs
    meta_train = pd.read_csv(Config.TRAIN_META)
    meta_val = pd.read_csv(Config.VAL_META)
    meta_test = pd.read_csv(Config.TEST_META)

    logger.info("Merging metadata with raw data...")

    # Helper to merge and clean
    def merge_data(meta_df, raw_df, is_test=False):
        # Merge on request_id
        merged = pd.merge(
            meta_df, raw_df, on=Config.ID_COL, how="left", suffixes=("", "_raw")
        )

        # If not test, ensure target column is integer and comes from metadata (ground truth)
        if not is_test:
            if Config.TARGET_COL in merged.columns:
                merged[Config.TARGET_COL] = merged[Config.TARGET_COL].astype(int)

        # Drop columns that might have been duplicated with _raw suffix if they exist
        cols_to_drop = [c for c in merged.columns if c.endswith("_raw")]
        if cols_to_drop:
            merged.drop(columns=cols_to_drop, inplace=True)

        return merged

    # The raw train file contains both train and val samples (original full train set)
    # We merge specific metadata splits with the full raw train dataframe
    df_train = merge_data(meta_train, df_raw_train, is_test=False)
    df_val = merge_data(meta_val, df_raw_train, is_test=False)

    # The raw test file corresponds to the test metadata
    df_test = merge_data(meta_test, df_raw_test, is_test=True)

    logger.info(
        f"Data loaded. Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
    )

    # 3. Save to Cache
    logger.info("Saving datasets to cache...")
    try:
        # Ensure directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        df_train.to_parquet(train_cache, index=False)
        df_val.to_parquet(val_cache, index=False)
        df_test.to_parquet(test_cache, index=False)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

    # 4. Handle Debug Mode
    if Config.DEBUG:
        logger.info(f"DEBUG mode enabled. Sampling {Config.DEBUG_SAMPLES} rows.")
        df_train = df_train.head(Config.DEBUG_SAMPLES)
        df_val = df_val.head(Config.DEBUG_SAMPLES)
        df_test = df_test.head(Config.DEBUG_SAMPLES)

    return df_train, df_val, df_test
