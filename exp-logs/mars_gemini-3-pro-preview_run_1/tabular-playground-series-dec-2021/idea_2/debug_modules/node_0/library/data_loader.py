import pandas as pd
from library.config import Config
from library.feature_engineering import process_and_cache_data


def load_and_process_data(load_cached_data: bool = True, max_samples: int = None):
    """
    Loads train, validation, and test datasets, applies feature engineering (via library),
    and prepares feature matrices (X) and target vectors (y).

    Args:
        load_cached_data (bool): Whether to load processed data from cache if available.
        max_samples (int, optional): If set, limits the number of samples for debugging.

    Returns:
        Tuple containing:
        - X_train (pd.DataFrame): Training features
        - y_train (pd.Series): Training target
        - X_val (pd.DataFrame): Validation features
        - y_val (pd.Series): Validation target
        - X_test (pd.DataFrame): Test features
        - test_ids (pd.Series): Test IDs for submission
    """
    # Load data using the library function which handles caching and feature engineering
    train_df, val_df, test_df = process_and_cache_data(
        load_cached_data=load_cached_data
    )

    # Debugging: Subsample if max_samples is provided
    if max_samples is not None:
        print(f"Debug mode: Limiting data to {max_samples} samples.")
        train_df = train_df.iloc[:max_samples]
        val_df = val_df.iloc[:max_samples]
        test_df = test_df.iloc[:max_samples]

    # Prepare Training Data
    # Drop Id and Target to get X
    # errors='ignore' ensures robustness if columns are missing or already dropped
    X_train = train_df.drop(columns=[Config.TARGET_COL, Config.ID_COL], errors="ignore")
    y_train = train_df[Config.TARGET_COL]

    # Prepare Validation Data
    X_val = val_df.drop(columns=[Config.TARGET_COL, Config.ID_COL], errors="ignore")
    y_val = val_df[Config.TARGET_COL]

    # Prepare Test Data
    # Test data does not have the Target column
    test_ids = test_df[Config.ID_COL]
    X_test = test_df.drop(columns=[Config.ID_COL], errors="ignore")

    return X_train, y_train, X_val, y_val, X_test, test_ids
