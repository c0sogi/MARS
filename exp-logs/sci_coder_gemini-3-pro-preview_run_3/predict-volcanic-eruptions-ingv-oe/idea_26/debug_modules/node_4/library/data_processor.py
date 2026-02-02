import pandas as pd
import numpy as np
import os
from library import config
from library import feature_engineering


def process_all_segments(metadata_path, split_name, load_cached_data=True):
    """
    Wrapper function to process segments using the feature_engineering library.
    This satisfies the requirement to implement 'process_all_segments' by delegating
    to the provided library which handles joblib parallelization and caching.
    """
    return feature_engineering.process_metadata_split(
        metadata_path, split_name, load_cached_data=load_cached_data
    )


def _prepare_dataset(df, is_test=False):
    """
    Internal helper to separate features, targets, and IDs from the raw dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing features and metadata.
        is_test (bool): Whether this is the test set (no target).

    Returns:
        tuple: (X, y) for train/val, or (X, segment_ids) for test.
    """
    if df is None or df.empty:
        print("Warning: Empty dataframe provided to _prepare_dataset.")
        return None, None

    # Define columns to exclude from features
    # 'file_path' might not be in the output of feature_engineering, but good to be safe
    # 'segment_id' and 'time_to_eruption' are metadata/target
    exclude_cols = ["segment_id", "time_to_eruption", "file_path"]

    # Select feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Create Feature Matrix X
    X = df[feature_cols].copy()

    if is_test:
        # For test set, return X and segment_ids (for submission)
        ids = df["segment_id"].copy()
        print(f"Prepared Test Data: X shape={X.shape}, ids shape={ids.shape}")
        return X, ids
    else:
        # For train/val sets, return X and y
        if "time_to_eruption" not in df.columns:
            raise ValueError(
                "Target column 'time_to_eruption' missing from training/validation data."
            )

        y = df["time_to_eruption"].copy()
        print(f"Prepared Train/Val Data: X shape={X.shape}, y shape={y.shape}")
        return X, y


def load_train_data(load_cached_data=True):
    """
    Loads the training dataset.

    Returns:
        tuple: (X_train, y_train)
    """
    print("Loading Training Data...")
    df = feature_engineering.get_train_data(load_cached_data=load_cached_data)
    return _prepare_dataset(df, is_test=False)


def load_val_data(load_cached_data=True):
    """
    Loads the validation dataset.

    Returns:
        tuple: (X_val, y_val)
    """
    print("Loading Validation Data...")
    df = feature_engineering.get_val_data(load_cached_data=load_cached_data)
    return _prepare_dataset(df, is_test=False)


def load_test_data(load_cached_data=True):
    """
    Loads the test dataset.

    Returns:
        tuple: (X_test, test_ids)
    """
    print("Loading Test Data...")
    df = feature_engineering.get_test_data(load_cached_data=load_cached_data)
    return _prepare_dataset(df, is_test=True)
