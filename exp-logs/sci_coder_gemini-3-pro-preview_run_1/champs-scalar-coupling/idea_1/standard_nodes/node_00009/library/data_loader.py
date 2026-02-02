import pandas as pd
from typing import Tuple, Optional
from library.config import Config
from library.features import process_data


def load_and_process_data(
    split_name: str, load_cached_data: bool = True, debug_nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Loads and processes data for a specific split (train, val, or test).

    This function acts as a wrapper around library.features.process_data,
    which handles file loading, structure merging, feature engineering,
    and caching to Parquet files.

    Args:
        split_name (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.
        debug_nrows (int, optional): If set, only loads this many rows for debugging.

    Returns:
        pd.DataFrame: The processed dataframe with all features and targets.
    """
    # Delegate to the centralized feature processing pipeline
    return process_data(
        split_name=split_name,
        load_cached_data=load_cached_data,
        debug_nrows=debug_nrows,
    )


def get_train_val_data(
    load_cached_data: bool = True, debug_nrows: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads both training and validation datasets and prepares them for model training.

    This function performs the following steps:
    1. Loads processed train and val DataFrames.
    2. Selects the specific features defined in Config.FEATURES.
    3. Separates the target variable.

    Args:
        load_cached_data (bool): If True, uses cached data if available.
        debug_nrows (int, optional): Limit rows for debugging.

    Returns:
        Tuple containing:
            - X_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training target values.
            - X_val (pd.DataFrame): Validation features.
            - y_val (pd.Series): Validation target values.
    """
    # Load processed full dataframes
    train_df = load_and_process_data("train", load_cached_data, debug_nrows)
    val_df = load_and_process_data("val", load_cached_data, debug_nrows)

    # Select features and target
    X_train = train_df[Config.FEATURES]
    y_train = train_df[Config.TARGET_COL]

    X_val = val_df[Config.FEATURES]
    y_val = val_df[Config.TARGET_COL]

    return X_train, y_train, X_val, y_val


def get_test_data(
    load_cached_data: bool = True, debug_nrows: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the test dataset and prepares it for prediction.

    Args:
        load_cached_data (bool): If True, uses cached data if available.
        debug_nrows (int, optional): Limit rows for debugging.

    Returns:
        Tuple containing:
            - X_test (pd.DataFrame): Test features.
            - ids (pd.Series): The 'id' column required for submission.
    """
    # Load processed test dataframe
    test_df = load_and_process_data("test", load_cached_data, debug_nrows)

    # Select features and ID
    X_test = test_df[Config.FEATURES]
    ids = test_df[Config.ID_COL]

    return X_test, ids
