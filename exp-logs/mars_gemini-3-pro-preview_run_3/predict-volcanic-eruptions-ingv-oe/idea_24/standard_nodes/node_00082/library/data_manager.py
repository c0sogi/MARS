import pandas as pd
import numpy as np
import os
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    DEBUG_SAMPLE_SIZE,
)
from library.feature_engineering import generate_features


def load_metadata(path):
    """
    Loads metadata from a CSV file.

    Args:
        path (str): Path to the metadata CSV.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")
    return pd.read_csv(path)


def build_feature_matrix(
    metadata_path, dataset_name, load_cached_data=True, debug_size=None
):
    """
    Constructs the feature matrix for a given dataset.
    Wraps the generate_features function from the feature_engineering library.

    Args:
        metadata_path (str): Path to metadata CSV.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to use cached parquet files.
        debug_size (int, optional): Number of rows to process for debugging.

    Returns:
        pd.DataFrame: DataFrame containing features and metadata.
    """
    return generate_features(
        metadata_path=metadata_path,
        dataset_name=dataset_name,
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )


def get_train_data(load_cached_data=True, debug_size=None):
    """
    Retrieves the training data (X, y).

    Args:
        load_cached_data (bool): Whether to use cached features.
        debug_size (int, optional): Limit size for debugging.

    Returns:
        tuple: (X (pd.DataFrame), y (pd.Series))
    """
    df = build_feature_matrix(
        TRAIN_META_PATH,
        "train",
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )

    # Separate target and features
    if "time_to_eruption" not in df.columns:
        raise ValueError("Training data must contain 'time_to_eruption' column.")

    y = df["time_to_eruption"]
    X = df.drop(columns=["segment_id", "time_to_eruption"])

    return X, y


def get_val_data(load_cached_data=True, debug_size=None):
    """
    Retrieves the validation data (X, y).

    Args:
        load_cached_data (bool): Whether to use cached features.
        debug_size (int, optional): Limit size for debugging.

    Returns:
        tuple: (X (pd.DataFrame), y (pd.Series))
    """
    df = build_feature_matrix(
        VAL_META_PATH, "val", load_cached_data=load_cached_data, debug_size=debug_size
    )

    if "time_to_eruption" not in df.columns:
        raise ValueError("Validation data must contain 'time_to_eruption' column.")

    y = df["time_to_eruption"]
    X = df.drop(columns=["segment_id", "time_to_eruption"])

    return X, y


def get_test_data(load_cached_data=True, debug_size=None):
    """
    Retrieves the test data (X, segment_ids).

    Args:
        load_cached_data (bool): Whether to use cached features.
        debug_size (int, optional): Limit size for debugging.

    Returns:
        tuple: (X (pd.DataFrame), segment_ids (pd.Series))
    """
    df = build_feature_matrix(
        TEST_META_PATH, "test", load_cached_data=load_cached_data, debug_size=debug_size
    )

    # Test data does not have target, but we need segment_id for submission
    segment_ids = df["segment_id"]

    # Drop segment_id to get feature matrix
    # Note: time_to_eruption should not be in test data, but we drop it if it exists just in case
    cols_to_drop = ["segment_id"]
    if "time_to_eruption" in df.columns:
        cols_to_drop.append("time_to_eruption")

    X = df.drop(columns=cols_to_drop)

    return X, segment_ids
