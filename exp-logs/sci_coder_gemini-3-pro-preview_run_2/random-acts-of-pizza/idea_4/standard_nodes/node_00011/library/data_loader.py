import pandas as pd
from library.config import Config
from library.utils import load_split_data


def load_datasets(logger=None, debug=False):
    """
    Loads the train, validation, and test datasets using the metadata and raw files.

    Args:
        logger (logging.Logger, optional): Logger instance for status updates.
        debug (bool): If True, returns a small subset of the data for debugging purposes.

    Returns:
        tuple: (df_train, df_val, df_test)
            - df_train (pd.DataFrame): Training data with labels.
            - df_val (pd.DataFrame): Validation data with labels.
            - df_test (pd.DataFrame): Test data without labels.
    """
    if logger:
        logger.info("Starting dataset loading process...")

    # Load Training Data
    df_train = load_split_data("train", logger=logger)

    # Load Validation Data
    df_val = load_split_data("val", logger=logger)

    # Load Test Data
    df_test = load_split_data("test", logger=logger)

    # Ensure target column is integer type for classification
    target_col = "requester_received_pizza"

    if target_col in df_train.columns:
        df_train[target_col] = df_train[target_col].astype(int)

    if target_col in df_val.columns:
        df_val[target_col] = df_val[target_col].astype(int)

    # Handle Debug Mode
    if debug:
        if logger:
            logger.info("Debug mode enabled: Subsampling datasets to 50 rows each.")
        df_train = df_train.head(50)
        df_val = df_val.head(50)
        df_test = df_test.head(50)

    if logger:
        logger.info(f"Data loading complete.")
        logger.info(f"Train shape: {df_train.shape}")
        logger.info(f"Val shape:   {df_val.shape}")
        logger.info(f"Test shape:  {df_test.shape}")

    return df_train, df_val, df_test
