import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    TARGET_COL,
    ID_COL,
)
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("data_loader")


def load_and_process_data(load_cached_data: bool = True):
    """
    Loads the dataset, merges train and validation sets into a union set,
    removes leakage features (retrieval-time stats), and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data
                                 from the cache directory.

    Returns:
        tuple: (train_df, test_df)
            - train_df (pd.DataFrame): The merged and cleaned training data.
            - test_df (pd.DataFrame): The test data.
    """
    # Define cache file paths
    train_cache_path = os.path.join(CACHE_DIR, "train_base.parquet")
    test_cache_path = os.path.join(CACHE_DIR, "test_base.parquet")

    # 1. Attempt to load from cache
    if load_cached_data:
        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            logger.info(f"Loading cached data from {CACHE_DIR}...")
            try:
                train_df = pd.read_parquet(train_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                logger.info(
                    f"Successfully loaded cached data. Train shape: {train_df.shape}, Test shape: {test_df.shape}"
                )
                return train_df, test_df
            except Exception as e:
                logger.warning(
                    f"Failed to load cached data: {e}. Proceeding to re-process."
                )
        else:
            logger.info("Cache files not found. Processing data from scratch...")
    else:
        logger.info("Ignoring cache. Processing data from scratch...")

    # 2. Load raw metadata (Parquet files)
    logger.info("Loading metadata files...")
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Train metadata not found at {TRAIN_METADATA_PATH}")
    if not os.path.exists(VAL_METADATA_PATH):
        raise FileNotFoundError(f"Validation metadata not found at {VAL_METADATA_PATH}")
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    df_train_part = pd.read_parquet(TRAIN_METADATA_PATH)
    df_val_part = pd.read_parquet(VAL_METADATA_PATH)
    df_test = pd.read_parquet(TEST_METADATA_PATH)

    # 3. Merge Train and Validation into Union Dataset
    logger.info("Merging Train and Validation sets into Union Dataset...")
    df_train = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

    # Verify merge
    expected_len = len(df_train_part) + len(df_val_part)
    if len(df_train) != expected_len:
        raise ValueError(
            f"Merge failed. Expected {expected_len} rows, got {len(df_train)}"
        )

    # 4. Leakage Prevention
    logger.info("Applying leakage prevention (dropping '_at_retrieval' columns)...")

    # Identify leakage columns in training data
    leakage_cols = [col for col in df_train.columns if col.endswith("_at_retrieval")]

    if leakage_cols:
        logger.info(f"Dropping {len(leakage_cols)} leakage columns: {leakage_cols}")
        df_train.drop(columns=leakage_cols, inplace=True)

    # Ensure target is integer
    if TARGET_COL in df_train.columns:
        df_train[TARGET_COL] = df_train[TARGET_COL].astype(int)
    else:
        raise ValueError(f"Target column '{TARGET_COL}' missing from training data.")

    # 5. Save to Cache
    logger.info(f"Saving processed data to cache: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    df_train.to_parquet(train_cache_path, index=False)
    df_test.to_parquet(test_cache_path, index=False)

    logger.info(
        f"Data processing complete. Union Train shape: {df_train.shape}, Test shape: {df_test.shape}"
    )

    return df_train, df_test
