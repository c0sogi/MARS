import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("data_loader")


def load_datasets(debug: bool = False):
    """
    Loads the train, validation, and test datasets from the metadata Parquet files.
    Separates the target variable from the features and extracts test IDs.

    Args:
        debug (bool): If True, loads only a small subset of the data (defined in Config.DEBUG_SAMPLES)
                      for debugging purposes.

    Returns:
        tuple: A tuple containing three tuples:
            - (X_train, y_train): Training features (DataFrame) and targets (Series).
            - (X_val, y_val): Validation features (DataFrame) and targets (Series).
            - (X_test, test_ids): Test features (DataFrame) and request IDs (Series).
    """
    logger.info("Loading datasets from metadata...")

    # Validate file existence
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(
            f"Training metadata file not found at {Config.TRAIN_PATH}"
        )
    if not os.path.exists(Config.VAL_PATH):
        raise FileNotFoundError(
            f"Validation metadata file not found at {Config.VAL_PATH}"
        )
    if not os.path.exists(Config.TEST_PATH):
        raise FileNotFoundError(f"Test metadata file not found at {Config.TEST_PATH}")

    # Load Parquet files
    # Parquet preserves data types, including lists for 'requester_subreddits_at_request'
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    logger.info(
        f"Original dataset shapes - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )

    # Apply Debug Sampling if requested
    if debug:
        logger.info(
            f"Debug mode enabled. Sampling {Config.DEBUG_SAMPLES} rows from each dataset."
        )
        train_df = train_df.iloc[: Config.DEBUG_SAMPLES].copy()
        val_df = val_df.iloc[: Config.DEBUG_SAMPLES].copy()
        test_df = test_df.iloc[: Config.DEBUG_SAMPLES].copy()

    # Process Training Data
    if Config.TARGET_COL not in train_df.columns:
        raise KeyError(
            f"Target column '{Config.TARGET_COL}' missing from training data."
        )
    y_train = train_df[Config.TARGET_COL].copy()
    X_train = train_df.drop(columns=[Config.TARGET_COL])

    # Process Validation Data
    if Config.TARGET_COL not in val_df.columns:
        raise KeyError(
            f"Target column '{Config.TARGET_COL}' missing from validation data."
        )
    y_val = val_df[Config.TARGET_COL].copy()
    X_val = val_df.drop(columns=[Config.TARGET_COL])

    # Process Test Data
    # Test data should not have the target, but we handle it safely if it exists (e.g. local testing)
    if Config.TARGET_COL in test_df.columns:
        X_test = test_df.drop(columns=[Config.TARGET_COL])
    else:
        X_test = test_df.copy()

    if Config.ID_COL not in test_df.columns:
        raise KeyError(f"ID column '{Config.ID_COL}' missing from test data.")
    test_ids = test_df[Config.ID_COL].copy()

    logger.info(
        f"Final feature shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}"
    )

    return (X_train, y_train), (X_val, y_val), (X_test, test_ids)
