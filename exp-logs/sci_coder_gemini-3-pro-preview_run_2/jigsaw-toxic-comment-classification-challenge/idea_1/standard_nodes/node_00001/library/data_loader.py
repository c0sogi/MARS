import os
import pandas as pd
from library.config import Config
from library.utils import setup_logger


def load_datasets(
    load_cached_data: bool = True,
    debug: bool = None,
    debug_sample_size: int = None,
):
    """
    Loads the training, validation, and test datasets.

    Implements a caching mechanism:
    1. If load_cached_data is True, attempts to load pre-processed Parquet files.
    2. If cache is missing or load_cached_data is False, loads from original CSVs,
       fills missing text values, and saves to Parquet cache.
    3. If debug is True, slices the datasets to the specified sample size.

    Args:
        load_cached_data (bool): Whether to attempt loading from the cache.
        debug (bool): Whether to run in debug mode (load subset).
        debug_sample_size (int): Number of rows to load in debug mode.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if debug is None:
        debug = Config.DEBUG
    if debug_sample_size is None:
        debug_sample_size = Config.DEBUG_SAMPLE_SIZE

    logger = setup_logger("data_loader")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    train_cache_path = os.path.join(Config.CACHE_DIR, "train.parquet")
    val_cache_path = os.path.join(Config.CACHE_DIR, "val.parquet")
    test_cache_path = os.path.join(Config.CACHE_DIR, "test.parquet")

    data_loaded = False

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):

            logger.info("Loading datasets from cache...")
            try:
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                data_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Reloading from source.")
                data_loaded = False
        else:
            logger.info("Cache files not found or incomplete.")

    # 2. If not loaded from cache, load from source CSVs and process
    if not data_loaded:
        logger.info("Loading datasets from metadata CSVs...")

        # Verify source files exist
        if not os.path.exists(Config.TRAIN_PATH):
            raise FileNotFoundError(f"Train file not found at {Config.TRAIN_PATH}")
        if not os.path.exists(Config.VAL_PATH):
            raise FileNotFoundError(f"Validation file not found at {Config.VAL_PATH}")
        if not os.path.exists(Config.TEST_PATH):
            raise FileNotFoundError(f"Test file not found at {Config.TEST_PATH}")

        # Load CSVs
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Preprocessing: Fill missing values in text columns
        logger.info("Preprocessing: Filling missing text values...")
        train_df["comment_text"] = train_df["comment_text"].fillna("")
        val_df["comment_text"] = val_df["comment_text"].fillna("")
        test_df["comment_text"] = test_df["comment_text"].fillna("")

        # Save to cache (save the full processed dataset)
        logger.info(f"Saving processed datasets to cache at {Config.CACHE_DIR}...")
        train_df.to_parquet(train_cache_path, index=False)
        val_df.to_parquet(val_cache_path, index=False)
        test_df.to_parquet(test_cache_path, index=False)

    # 3. Apply Debug Sampling if requested
    # We apply this AFTER loading/caching so the cache always contains the full dataset
    if debug:
        logger.info(f"Debug mode enabled. Sampling first {debug_sample_size} rows.")
        train_df = train_df.iloc[:debug_sample_size].copy()
        val_df = val_df.iloc[:debug_sample_size].copy()
        test_df = test_df.iloc[:debug_sample_size].copy()

    logger.info(f"Data Loaded Successfully.")
    logger.info(f"Train Shape: {train_df.shape}")
    logger.info(f"Val Shape:   {val_df.shape}")
    logger.info(f"Test Shape:  {test_df.shape}")

    return train_df, val_df, test_df
