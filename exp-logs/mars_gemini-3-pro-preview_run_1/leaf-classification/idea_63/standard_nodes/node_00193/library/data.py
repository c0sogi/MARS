import numpy as np
import pandas as pd
from library.config import TRAIN_META_PATH, VAL_META_PATH, TEST_META_PATH
from library.features import process_dataset


def get_data(load_cached_data: bool = True, debug_size: int = None):
    """
    Loads the training, validation, and test datasets.
    Uses the features module to extract geometric features and merge them with tabular data.
    Returns the data in a format suitable for the Custom Linear Discriminant model (float64).

    Args:
        load_cached_data (bool): Whether to use cached feature files. Defaults to True.
        debug_size (int, optional): If set, truncates the datasets to this number of samples for debugging.

    Returns:
        dict: A dictionary containing:
            - X_train (pd.DataFrame): Training features (float64).
            - y_train (np.ndarray): Training labels (strings).
            - X_val (pd.DataFrame): Validation features (float64).
            - y_val (np.ndarray): Validation labels (strings).
            - X_test (pd.DataFrame): Test features (float64).
            - test_ids (np.ndarray): Test image IDs.
    """

    # 1. Load and assemble datasets using the features library
    # This handles feature extraction, merging, alphanumeric column sorting, and caching.
    df_train = process_dataset(TRAIN_META_PATH, load_cached_data=load_cached_data)
    df_val = process_dataset(VAL_META_PATH, load_cached_data=load_cached_data)
    df_test = process_dataset(TEST_META_PATH, load_cached_data=load_cached_data)

    # 2. Apply debugging truncation if requested
    if debug_size is not None:
        df_train = df_train.iloc[:debug_size]
        df_val = df_val.iloc[:debug_size]
        df_test = df_test.iloc[:debug_size]

    # 3. Separate Features and Targets
    # Train
    # 'species' is the target, 'id' is metadata
    y_train = df_train["species"].values
    X_train = df_train.drop(columns=["id", "species"])

    # Val
    y_val = df_val["species"].values
    X_val = df_val.drop(columns=["id", "species"])

    # Test
    # Test set does not have 'species' column in the provided metadata/test.csv,
    # but we handle it safely if it were present (e.g. in a local validation scenario).
    test_ids = df_test["id"].values
    X_test = df_test.drop(columns=["id"])
    if "species" in X_test.columns:
        X_test = X_test.drop(columns=["species"])

    # 4. Enforce float64 precision
    # The model requires exact analytical inference, so we ensure high precision.
    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # 5. Verify Column Alignment
    # The features.process_dataset function enforces alphanumeric sorting,
    # but we explicitly validate that the feature space is consistent across splits.
    train_cols = list(X_train.columns)
    val_cols = list(X_val.columns)
    test_cols = list(X_test.columns)

    assert (
        train_cols == val_cols
    ), "Feature columns mismatch between Train and Val sets."
    assert (
        train_cols == test_cols
    ), "Feature columns mismatch between Train and Test sets."

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "test_ids": test_ids,
    }
