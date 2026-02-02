import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, setup_logger

# Initialize Logger
logger = setup_logger("data_factory")


def load_dataset(file_path):
    """
    Helper function to load a dataset from a parquet file.

    Args:
        file_path (str): Path to the parquet file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    return pd.read_parquet(file_path)


def prepare_datasets(load_cached_data=True, debug=False, debug_size=100):
    """
    Loads, merges, and prepares the datasets for the pipeline.

    Implements caching to speed up subsequent runs.
    Merges Train and Validation sets into a single Union Dataset for CV.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, downsamples the dataset for debugging.
        debug_size (int): Number of samples to keep if debug is True.

    Returns:
        tuple: (union_train_df, test_df)
    """
    set_seed(Config.RANDOM_STATE)

    # Define cache paths
    cache_train_path = os.path.join(Config.CACHE_DIR, "union_train.parquet")
    cache_test_path = os.path.join(Config.CACHE_DIR, "test_base.parquet")

    union_train_df = None
    test_df = None
    loaded_from_cache = False

    # 1. Try Loading from Cache
    if load_cached_data:
        if os.path.exists(cache_train_path) and os.path.exists(cache_test_path):
            try:
                logger.info(f"Loading datasets from cache: {Config.CACHE_DIR}")
                union_train_df = pd.read_parquet(cache_train_path)
                test_df = pd.read_parquet(cache_test_path)
                loaded_from_cache = True
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")
        else:
            logger.info("Cache files not found. Processing from scratch...")

    # 2. Compute/Process if not loaded from cache
    if not loaded_from_cache:
        logger.info("Loading raw metadata files...")

        # Load individual splits using Config paths
        train_df = load_dataset(Config.TRAIN_PATH)
        val_df = load_dataset(Config.VAL_PATH)
        test_df = load_dataset(Config.TEST_PATH)

        logger.info(f"Raw Train shape: {train_df.shape}")
        logger.info(f"Raw Val shape: {val_df.shape}")
        logger.info(f"Raw Test shape: {test_df.shape}")

        # Merge Train and Val into Union Train
        logger.info("Merging Train and Validation sets into Union Dataset...")
        union_train_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

        # Save to cache
        logger.info(f"Saving processed datasets to cache: {Config.CACHE_DIR}")
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        union_train_df.to_parquet(cache_train_path, index=False)
        test_df.to_parquet(cache_test_path, index=False)

    # 3. Handle Debugging (Downsampling)
    # This is done AFTER caching to ensure the cache contains the full dataset
    if debug:
        logger.info(f"Debug mode enabled. Sampling {debug_size} rows...")
        if len(union_train_df) > debug_size:
            union_train_df = union_train_df.sample(
                n=debug_size, random_state=Config.RANDOM_STATE
            ).reset_index(drop=True)
        if len(test_df) > debug_size:
            test_df = test_df.sample(
                n=debug_size, random_state=Config.RANDOM_STATE
            ).reset_index(drop=True)

    logger.info(f"Final Union Train shape: {union_train_df.shape}")
    logger.info(f"Final Test shape: {test_df.shape}")

    return union_train_df, test_df
