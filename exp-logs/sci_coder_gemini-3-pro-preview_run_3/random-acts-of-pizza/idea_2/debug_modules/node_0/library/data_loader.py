import os
import pandas as pd
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("data_loader")


def load_data_splits(max_samples: int = None):
    """
    Loads the training, validation, and test datasets from the metadata Parquet files.

    Args:
        max_samples (int, optional): If provided, limits the number of samples loaded
                                     for each split. Useful for debugging or quick testing.

    Returns:
        tuple: A tuple containing three pandas DataFrames: (train_df, val_df, test_df).

    Raises:
        FileNotFoundError: If any of the data files specified in Config do not exist.
    """
    logger.info("Loading data splits from metadata...")

    # Define paths from Config
    train_path = Config.TRAIN_DATA_PATH
    val_path = Config.VAL_DATA_PATH
    test_path = Config.TEST_DATA_PATH

    # Check existence
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            error_msg = f"Data file not found at: {path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    # Load DataFrames
    try:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
    except Exception as e:
        logger.error(f"Failed to read Parquet files: {e}")
        raise e

    # Apply subsampling if max_samples is specified
    if max_samples is not None:
        logger.info(
            f"Subsampling data to {max_samples} samples per split for debugging."
        )
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max_samples)
        test_df = test_df.head(max_samples)

    # Log dataset shapes
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Val shape:   {val_df.shape}")
    logger.info(f"Test shape:  {test_df.shape}")

    logger.info("Data loading complete.")

    return train_df, val_df, test_df
