import os
import pandas as pd
from library.config import TRAIN_PATH, VAL_PATH, TEST_PATH, WORKING_DIR, TEXT_COLS
from library.utils import get_logger, load_cache, save_cache

# Initialize Logger
logger = get_logger("data_factory")


def clean_text(text):
    """
    Performs basic string sanitization.
    Ensures the input is a string and handles missing values.

    Args:
        text: Input text (string, float, None, etc.)

    Returns:
        str: Sanitized string.
    """
    if pd.isna(text):
        return ""
    return str(text).strip()


def load_union_dataset(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads the training and validation datasets, merges them into a single
    'Union Dataset', and performs basic text cleaning.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The combined training and validation data.
    """
    cache_filename = "union_dataset.parquet"

    # 1. Try to load from cache
    if load_cached_data:
        cached_df = load_cache(cache_filename, WORKING_DIR)
        if cached_df is not None:
            logger.info(f"Loaded Union Dataset from cache: {cached_df.shape}")
            return cached_df

    logger.info("Generating Union Dataset from scratch...")

    # 2. Load raw metadata files
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
        raise FileNotFoundError(
            f"Metadata files not found. Expected at {TRAIN_PATH} and {VAL_PATH}"
        )

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    # 3. Merge into Union Dataset
    # We reset index to ensure a continuous index across the union
    union_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    # 4. Basic Text Cleaning
    # Ensure text columns defined in config are strings and not NaN
    for col in TEXT_COLS:
        if col in union_df.columns:
            union_df[col] = union_df[col].apply(clean_text)

    logger.info(f"Union Dataset created. Shape: {union_df.shape}")

    # 5. Save to cache
    save_cache(union_df, cache_filename, WORKING_DIR)
    logger.info(
        f"Union Dataset saved to cache at {os.path.join(WORKING_DIR, cache_filename)}"
    )

    return union_df


def load_test_dataset(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads the test dataset and performs basic text cleaning.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The test data.
    """
    cache_filename = "test_dataset.parquet"

    # 1. Try to load from cache
    if load_cached_data:
        cached_df = load_cache(cache_filename, WORKING_DIR)
        if cached_df is not None:
            logger.info(f"Loaded Test Dataset from cache: {cached_df.shape}")
            return cached_df

    logger.info("Loading Test Dataset from scratch...")

    # 2. Load raw metadata file
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Metadata file not found. Expected at {TEST_PATH}")

    test_df = pd.read_parquet(TEST_PATH)

    # 3. Basic Text Cleaning
    for col in TEXT_COLS:
        if col in test_df.columns:
            test_df[col] = test_df[col].apply(clean_text)

    logger.info(f"Test Dataset loaded. Shape: {test_df.shape}")

    # 4. Save to cache
    save_cache(test_df, cache_filename, WORKING_DIR)
    logger.info(
        f"Test Dataset saved to cache at {os.path.join(WORKING_DIR, cache_filename)}"
    )

    return test_df
