import os
import pandas as pd
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("data_loader")


def load_union_dataset(load_cached_data: bool = True, debug_size: int = None):
    """
    Loads the training, validation, and test datasets. Merges training and validation
    into a single 'Union Dataset' for the Hept-View Stacking Ensemble.

    Implements deterministic caching for the union dataset.

    Args:
        load_cached_data (bool): If True, attempts to load pre-merged data from disk.
        debug_size (int, optional): If provided, downsamples the data for debugging.

    Returns:
        tuple: (train_df, test_df)
            - train_df: The merged training and validation data (Union Dataset).
            - test_df: The test data.
    """
    # Define cache path for the union dataset
    cache_path = os.path.join(Config.WORKING_DIR, "union_train.parquet")

    train_df = None

    # 1. Attempt to load from cache
    if load_cached_data:
        if os.path.exists(cache_path):
            logger.info(f"Loading cached union dataset from {cache_path}")
            try:
                train_df = pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")
        else:
            logger.info("Cache not found. Generating union dataset...")

    # 2. Compute from scratch if not loaded
    if train_df is None:
        logger.info(f"Loading training data from {Config.TRAIN_PATH}")
        df_train_part = pd.read_parquet(Config.TRAIN_PATH)

        logger.info(f"Loading validation data from {Config.VAL_PATH}")
        df_val_part = pd.read_parquet(Config.VAL_PATH)

        # Merge to create Union Dataset
        logger.info("Merging Train and Validation sets into Union Dataset...")
        train_df = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

        # Shuffle the union dataset to mix train and val samples (reproducibly)
        train_df = train_df.sample(
            frac=1.0, random_state=Config.RANDOM_STATE
        ).reset_index(drop=True)

        # Save to cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            train_df.to_parquet(cache_path, index=False)
            logger.info(f"Saved union dataset to cache at {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    # 3. Load Test Data
    logger.info(f"Loading test data from {Config.TEST_PATH}")
    test_df = pd.read_parquet(Config.TEST_PATH)

    # 4. Apply Debugging Sampling (if requested)
    if debug_size is not None and debug_size > 0:
        logger.info(f"Debug mode enabled. Sampling {debug_size} rows.")
        if len(train_df) > debug_size:
            train_df = train_df.iloc[:debug_size].copy()
        if len(test_df) > debug_size:
            test_df = test_df.iloc[:debug_size].copy()

    # Log final shapes
    logger.info(f"Final Union Train Shape: {train_df.shape}")
    logger.info(f"Final Test Shape: {test_df.shape}")

    return train_df, test_df
