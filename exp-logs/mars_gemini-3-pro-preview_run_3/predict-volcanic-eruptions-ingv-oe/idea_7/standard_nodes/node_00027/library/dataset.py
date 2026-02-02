import os
import pandas as pd
import numpy as np
from library.feature_extraction import generate_features

# Configuration Constants
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")


def get_train_data(debug_size=None, load_cached_data=True):
    """
    Loads the training dataset.

    Args:
        debug_size (int, optional): Number of segments to process for debugging.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (X, y) where X is a pandas DataFrame of features and y is a numpy array of targets.
    """
    save_name = "train_features"
    if debug_size is not None:
        save_name = f"{save_name}_debug_{debug_size}"

    # Generate or load features using the provided library
    df = generate_features(
        metadata_path=TRAIN_META_PATH,
        load_cached_data=load_cached_data,
        save_name=save_name,
        debug_size=debug_size,
    )

    # Separate features and target
    # The library function ensures 'time_to_eruption' is merged if present in metadata
    y = df["time_to_eruption"].values
    X = df.drop(columns=["segment_id", "time_to_eruption"])

    return X, y


def get_val_data(debug_size=None, load_cached_data=True):
    """
    Loads the validation dataset.

    Args:
        debug_size (int, optional): Number of segments to process for debugging.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (X, y) where X is a pandas DataFrame of features and y is a numpy array of targets.
    """
    save_name = "val_features"
    if debug_size is not None:
        save_name = f"{save_name}_debug_{debug_size}"

    df = generate_features(
        metadata_path=VAL_META_PATH,
        load_cached_data=load_cached_data,
        save_name=save_name,
        debug_size=debug_size,
    )

    y = df["time_to_eruption"].values
    X = df.drop(columns=["segment_id", "time_to_eruption"])

    return X, y


def get_test_data(debug_size=None, load_cached_data=True):
    """
    Loads the test dataset.

    Args:
        debug_size (int, optional): Number of segments to process for debugging.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (X, ids) where X is a pandas DataFrame of features and ids is a numpy array of segment_ids.
    """
    save_name = "test_features"
    if debug_size is not None:
        save_name = f"{save_name}_debug_{debug_size}"

    df = generate_features(
        metadata_path=TEST_META_PATH,
        load_cached_data=load_cached_data,
        save_name=save_name,
        debug_size=debug_size,
    )

    # For test set, we need segment_id for the submission file
    ids = df["segment_id"].values
    X = df.drop(columns=["segment_id"])

    # Safety: ensure target column is removed if it somehow exists (unlikely for test)
    if "time_to_eruption" in X.columns:
        X = X.drop(columns=["time_to_eruption"])

    return X, ids
