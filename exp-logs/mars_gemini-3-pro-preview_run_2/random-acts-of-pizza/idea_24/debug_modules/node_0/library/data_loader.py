import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger


def load_dataset(load_cached_data: bool = True):
    """
    Loads the dataset, merging raw JSON data with metadata splits.
    Implements caching using Parquet files.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    logger = setup_logger("data_loader")

    # Define cache paths
    cache_train = Config.CACHE_TRAIN_FEATURES
    cache_val = Config.CACHE_VAL_FEATURES
    cache_test = Config.CACHE_TEST_FEATURES

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            logger.info("Loading datasets from cache...")
            try:
                df_train = pd.read_parquet(cache_train)
                df_val = pd.read_parquet(cache_val)
                df_test = pd.read_parquet(cache_test)

                # Handle debug sampling on cached data if requested
                if Config.DEBUG_SAMPLE_SIZE:
                    logger.info(
                        f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows."
                    )
                    df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
                    df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
                    df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

                return df_train, df_val, df_test
            except Exception as e:
                logger.warning(
                    f"Failed to load cache: {e}. Proceeding to reload raw data."
                )
        else:
            logger.info("Cache files not found. Loading raw data...")

    # 2. Load Raw Data
    logger.info("Loading raw JSON files...")
    with open(Config.TRAIN_JSON, "r") as f:
        raw_train_data = json.load(f)

    with open(Config.TEST_JSON, "r") as f:
        raw_test_data = json.load(f)

    # 3. Load Metadata
    logger.info("Loading metadata splits...")
    meta_train = pd.read_csv(Config.TRAIN_META_PATH)
    meta_val = pd.read_csv(Config.VAL_META_PATH)
    meta_test = pd.read_csv(Config.TEST_META_PATH)

    # 4. Helper to merge metadata with raw data
    def _create_dataframe(meta_df, raw_data_source):
        # Use sample_index to fetch data efficiently
        indices = meta_df["sample_index"].values
        records = [raw_data_source[i] for i in indices]

        df = pd.DataFrame(records)

        # Ensure alignment and keep metadata columns if needed (like target from metadata)
        # We trust the raw data for features, but metadata for labels/splits
        # However, raw train data contains the label 'requester_received_pizza'
        # Metadata also contains 'requester_received_pizza'

        # If metadata has the target, we prioritize/ensure it matches or use it
        if Config.TARGET_COL in meta_df.columns:
            df[Config.TARGET_COL] = meta_df[Config.TARGET_COL].values

        return df

    logger.info("Constructing DataFrames...")
    df_train = _create_dataframe(meta_train, raw_train_data)
    df_val = _create_dataframe(meta_val, raw_train_data)
    df_test = _create_dataframe(meta_test, raw_test_data)

    # 5. Basic Preprocessing / Type Enforcement

    # Ensure target is int
    if Config.TARGET_COL in df_train.columns:
        df_train[Config.TARGET_COL] = df_train[Config.TARGET_COL].astype(int)
    if Config.TARGET_COL in df_val.columns:
        df_val[Config.TARGET_COL] = df_val[Config.TARGET_COL].astype(int)

    # Ensure text columns are strings and handle NaNs
    for col in Config.TEXT_COLS:
        for df in [df_train, df_val, df_test]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
            else:
                # Create empty column if missing (though unlikely based on schema)
                df[col] = ""

    # 6. Debug Sampling (Before saving to cache? No, usually we cache full data and sample on load
    #    But if we are processing from scratch, we might want to save the full processed data first)

    # Save full processed data to cache
    logger.info("Saving processed datasets to cache...")
    Config.setup()  # Ensure directory exists

    # We save the full datasets to cache so subsequent runs can sample from them if needed
    try:
        df_train.to_parquet(cache_train, index=False)
        df_val.to_parquet(cache_val, index=False)
        df_test.to_parquet(cache_test, index=False)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    # Apply Debug Sampling if requested
    if Config.DEBUG_SAMPLE_SIZE:
        logger.info(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    logger.info(
        f"Data loaded. Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
    )
    return df_train, df_val, df_test
