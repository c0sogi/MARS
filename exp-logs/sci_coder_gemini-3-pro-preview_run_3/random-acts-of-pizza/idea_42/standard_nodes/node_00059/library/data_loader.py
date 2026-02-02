import os
import pandas as pd
import numpy as np
from library import config, utils

# Initialize logger
logger = utils.setup_logging("data_loader")


def _clean_text(series):
    """
    Cleans a pandas Series of text by filling NaNs and converting to string.
    """
    return series.fillna("").astype(str)


def _preprocess_dataframe(df, is_test=False):
    """
    Applies cleaning rules, leakage prevention, and basic text preparation.

    Args:
        df (pd.DataFrame): The dataframe to process.
        is_test (bool): Whether this is the test set.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # 1. Leakage Prevention: Drop columns suffixed with '_at_retrieval'
    # These columns contain future information not available at prediction time.
    retrieval_cols = [c for c in df.columns if c.endswith("_at_retrieval")]
    if retrieval_cols:
        logger.info(
            f"Dropping {len(retrieval_cols)} leakage columns ending in '_at_retrieval'"
        )
        df = df.drop(columns=retrieval_cols)

    # 2. Text Handling
    # Ensure we use 'request_text_edit_aware' as the primary body text
    # If it doesn't exist (unlikely given metadata), fallback to 'request_text'
    if "request_text_edit_aware" not in df.columns and "request_text" in df.columns:
        df["request_text_edit_aware"] = df["request_text"]

    # Fill NaNs
    df["request_title"] = _clean_text(
        df.get("request_title", pd.Series(dtype="object"))
    )
    df["request_text_edit_aware"] = _clean_text(
        df.get("request_text_edit_aware", pd.Series(dtype="object"))
    )

    # 3. Text Concatenation (as per Idea description)
    # Join Title and Body for downstream NLP tasks
    df["text_combined"] = df["request_title"] + " " + df["request_text_edit_aware"]

    # 4. Target Standardization (only for train/val)
    if not is_test and "requester_received_pizza" in df.columns:
        df["requester_received_pizza"] = df["requester_received_pizza"].astype(int)

    return df


def load_dataset(mode="train", load_cached_data=True, debug_size=None):
    """
    Loads the dataset (train/val or test), applying preprocessing and caching.

    Args:
        mode (str): 'train' to load training and validation sets, 'test' for test set.
        load_cached_data (bool): If True, attempts to load from cache first.
        debug_size (int, optional): If set, limits the number of rows returned for debugging.

    Returns:
        tuple or pd.DataFrame:
            - If mode='train': returns (train_df, val_df)
            - If mode='test': returns test_df
    """
    utils.set_seed(config.SEED)

    # Define Cache Paths
    # We use the CACHE_DIR from config which maps to ./working/idea_42/cache
    cache_dir = config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    if mode == "train":
        train_cache_path = os.path.join(cache_dir, "train_processed.parquet")
        val_cache_path = os.path.join(cache_dir, "val_processed.parquet")

        # Attempt to load from cache
        if (
            load_cached_data
            and os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
        ):
            logger.info(f"Loading processed train/val data from cache: {cache_dir}")
            train_df = utils.load_data_cache(train_cache_path)
            val_df = utils.load_data_cache(val_cache_path)
        else:
            logger.info("Processing train/val data from raw metadata...")
            # Load from Metadata
            train_df = pd.read_parquet(config.TRAIN_METADATA_PATH)
            val_df = pd.read_parquet(config.VAL_METADATA_PATH)

            # Process
            train_df = _preprocess_dataframe(train_df, is_test=False)
            val_df = _preprocess_dataframe(val_df, is_test=False)

            # Save to Cache
            logger.info(f"Saving processed train/val data to cache: {cache_dir}")
            utils.save_data_cache(train_df, train_cache_path)
            utils.save_data_cache(val_df, val_cache_path)

        # Apply Debug Slicing
        if debug_size is not None:
            logger.info(f"Debug mode: Slicing train/val to {debug_size} samples")
            train_df = train_df.head(debug_size)
            val_df = val_df.head(debug_size)

        logger.info(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
        return train_df, val_df

    elif mode == "test":
        test_cache_path = os.path.join(cache_dir, "test_processed.parquet")

        # Attempt to load from cache
        if load_cached_data and os.path.exists(test_cache_path):
            logger.info(f"Loading processed test data from cache: {cache_dir}")
            test_df = utils.load_data_cache(test_cache_path)
        else:
            logger.info("Processing test data from raw metadata...")
            # Load from Metadata
            test_df = pd.read_parquet(config.TEST_METADATA_PATH)

            # Process
            test_df = _preprocess_dataframe(test_df, is_test=True)

            # Save to Cache
            logger.info(f"Saving processed test data to cache: {cache_dir}")
            utils.save_data_cache(test_df, test_cache_path)

        # Apply Debug Slicing
        if debug_size is not None:
            logger.info(f"Debug mode: Slicing test to {debug_size} samples")
            test_df = test_df.head(debug_size)

        logger.info(f"Test shape: {test_df.shape}")
        return test_df

    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'train' or 'test'.")
