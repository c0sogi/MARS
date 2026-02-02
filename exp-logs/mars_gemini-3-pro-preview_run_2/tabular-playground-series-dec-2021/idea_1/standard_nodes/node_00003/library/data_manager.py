import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("data_manager")


class LabelMapper:
    """
    Handles encoding of target labels to 0-indexed integers and decoding back to original class labels.
    """

    @staticmethod
    def encode(y):
        """
        Maps original class labels to 0-indexed integers.
        Args:
            y (array-like): Original target values.
        Returns:
            np.ndarray: Encoded target values.
        """
        # Ensure input is a Series for mapping
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        return y.map(Config.TARGET_MAPPING).fillna(-1).astype(int).values

    @staticmethod
    def decode(y_encoded):
        """
        Maps 0-indexed integers back to original class labels.
        Args:
            y_encoded (array-like): Encoded target values.
        Returns:
            np.ndarray: Original target values.
        """
        if not isinstance(y_encoded, pd.Series):
            y_encoded = pd.Series(y_encoded)
        return (
            y_encoded.map(Config.INVERSE_TARGET_MAPPING).fillna(-1).astype(int).values
        )


def load_dataset(load_cached_data=True, debug=Config.DEBUG):
    """
    Loads the dataset, performing preprocessing and caching.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
        debug (bool): If True, loads a smaller subset of data for debugging.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache filenames based on debug mode
    prefix = "debug_" if debug else ""

    cache_paths = {
        "X_train": os.path.join(Config.WORKING_DIR, f"{prefix}X_train.parquet"),
        "y_train": os.path.join(Config.WORKING_DIR, f"{prefix}y_train.npy"),
        "X_val": os.path.join(Config.WORKING_DIR, f"{prefix}X_val.parquet"),
        "y_val": os.path.join(Config.WORKING_DIR, f"{prefix}y_val.npy"),
        "X_test": os.path.join(Config.WORKING_DIR, f"{prefix}X_test.parquet"),
        "test_ids": os.path.join(Config.WORKING_DIR, f"{prefix}test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_paths.values())

    if load_cached_data and cache_exists:
        logger.info(f"Loading cached dataset (debug={debug})...")
        X_train = pd.read_parquet(cache_paths["X_train"])
        y_train = np.load(cache_paths["y_train"])
        X_val = pd.read_parquet(cache_paths["X_val"])
        y_val = np.load(cache_paths["y_val"])
        X_test = pd.read_parquet(cache_paths["X_test"])
        test_ids = np.load(cache_paths["test_ids"])

        logger.info(f"Loaded train shape: {X_train.shape}")
        return X_train, y_train, X_val, y_val, X_test, test_ids

    # If cache miss or forced reload, process from metadata
    logger.info(f"Processing dataset from metadata (debug={debug})...")

    # Load raw metadata
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # Apply debug sampling if requested
    if debug:
        logger.info(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows...")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Process Training Data
    # Drop Id, separate Target
    if Config.ID_COL in df_train.columns:
        df_train = df_train.drop(columns=[Config.ID_COL])

    y_train_raw = df_train[Config.TARGET_COL]
    X_train = df_train.drop(columns=[Config.TARGET_COL])
    y_train = LabelMapper.encode(y_train_raw)

    # Process Validation Data
    if Config.ID_COL in df_val.columns:
        df_val = df_val.drop(columns=[Config.ID_COL])

    y_val_raw = df_val[Config.TARGET_COL]
    X_val = df_val.drop(columns=[Config.TARGET_COL])
    y_val = LabelMapper.encode(y_val_raw)

    # Process Test Data
    # Extract IDs for submission, then drop Id column
    test_ids = df_test[Config.ID_COL].values
    X_test = df_test.drop(columns=[Config.ID_COL])
    # Note: Test data does not have target column

    # Save to cache
    logger.info("Saving processed data to cache...")
    X_train.to_parquet(cache_paths["X_train"], index=False)
    np.save(cache_paths["y_train"], y_train)

    X_val.to_parquet(cache_paths["X_val"], index=False)
    np.save(cache_paths["y_val"], y_val)

    X_test.to_parquet(cache_paths["X_test"], index=False)
    np.save(cache_paths["test_ids"], test_ids)

    logger.info(f"Processing complete. Train shape: {X_train.shape}")

    return X_train, y_train, X_val, y_val, X_test, test_ids
