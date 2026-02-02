import os
import json
import pandas as pd
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("data_loader")


def load_raw_json(file_path):
    """
    Helper function to load raw JSON data into a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def load_dataset(load_cached_data: bool = True):
    """
    Loads the training, validation, and test datasets.

    Implements caching using Parquet format. If cached files exist and
    load_cached_data is True, loads from disk. Otherwise, reads raw JSON
    and metadata CSVs, merges them, saves to cache, and returns DataFrames.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Define cache paths
    cache_dir = Config.CACHE_DIR
    Config.ensure_directories()

    train_cache_path = os.path.join(cache_dir, "train_merged.parquet")
    val_cache_path = os.path.join(cache_dir, "val_merged.parquet")
    test_cache_path = os.path.join(cache_dir, "test_merged.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    )

    if load_cached_data and cache_exists:
        logger.info("Loading datasets from cache...")
        try:
            df_train = pd.read_parquet(train_cache_path)
            df_val = pd.read_parquet(val_cache_path)
            df_test = pd.read_parquet(test_cache_path)

            if Config.DEBUG:
                logger.info(
                    f"DEBUG mode enabled. Slicing datasets to {Config.DEBUG_SAMPLE_SIZE} samples."
                )
                df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
                df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
                df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

            return df_train, df_val, df_test
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reloading from raw sources.")

    logger.info("Loading raw data and metadata...")

    # Load Metadata
    if not os.path.exists(Config.TRAIN_META):
        raise FileNotFoundError(f"Metadata file missing: {Config.TRAIN_META}")

    meta_train = pd.read_csv(Config.TRAIN_META)
    meta_val = pd.read_csv(Config.VAL_META)
    meta_test = pd.read_csv(Config.TEST_META)

    # Load Raw Data
    # Note: Train and Val come from train.json, Test comes from test.json
    raw_train_full = load_raw_json(Config.TRAIN_JSON)
    raw_test_full = load_raw_json(Config.TEST_JSON)

    logger.info("Merging metadata with raw features...")

    # Process Train
    # We drop the target from raw if it exists to avoid duplication/suffixes during merge,
    # relying on the target provided in the metadata.
    cols_to_exclude_merge = (
        [Config.TARGET_COL] if Config.TARGET_COL in raw_train_full.columns else []
    )

    df_train = meta_train.merge(
        raw_train_full.drop(columns=cols_to_exclude_merge, errors="ignore"),
        on=Config.ID_COL,
        how="left",
    )

    # Process Validation
    df_val = meta_val.merge(
        raw_train_full.drop(columns=cols_to_exclude_merge, errors="ignore"),
        on=Config.ID_COL,
        how="left",
    )

    # Process Test
    # Test data usually doesn't have the target, but we check just in case
    cols_to_exclude_test = (
        [Config.TARGET_COL] if Config.TARGET_COL in raw_test_full.columns else []
    )

    df_test = meta_test.merge(
        raw_test_full.drop(columns=cols_to_exclude_test, errors="ignore"),
        on=Config.ID_COL,
        how="left",
    )

    # Save to Cache
    logger.info(f"Saving processed datasets to {cache_dir}...")
    try:
        df_train.to_parquet(train_cache_path, index=False)
        df_val.to_parquet(val_cache_path, index=False)
        df_test.to_parquet(test_cache_path, index=False)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

    # Handle Debug Mode
    if Config.DEBUG:
        logger.info(
            f"DEBUG mode enabled. Slicing datasets to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    logger.info(
        f"Data loading complete. Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
    )
    return df_train, df_val, df_test
