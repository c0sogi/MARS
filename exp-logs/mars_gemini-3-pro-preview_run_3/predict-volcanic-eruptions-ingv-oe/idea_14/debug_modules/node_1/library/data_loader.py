import pandas as pd
from library.config import Config
from library.feature_engineering import process_dataset


def generate_dataset(metadata_path, output_filename, load_cached_data=True, debug=None):
    """
    Generates the dataset (X, y) from metadata using the feature engineering pipeline.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        output_filename (str): Filename for the cached parquet file.
        load_cached_data (bool): Whether to load from cache if available.
        debug (bool, optional): If True, runs in debug mode (small sample).
                                If None, uses Config.DEBUG.

    Returns:
        tuple: (X, y) where X is the feature DataFrame and y is the target Series (or None).
               The index of X and y is set to 'segment_id'.
    """
    # Manage global debug state for the feature engineering module
    original_debug = Config.DEBUG
    if debug is not None:
        Config.DEBUG = debug

    try:
        # Delegate to the provided feature engineering pipeline
        # process_dataset handles loading metadata, parallel extraction, and caching
        df = process_dataset(
            metadata_path, output_filename, load_cached_data=load_cached_data
        )
    finally:
        # Restore original debug state
        if debug is not None:
            Config.DEBUG = original_debug

    # Set segment_id as index to preserve it for submission/tracking but remove from features
    if "segment_id" in df.columns:
        df = df.set_index("segment_id")

    # Separate target variable if present
    if "time_to_eruption" in df.columns:
        y = df["time_to_eruption"]
        X = df.drop(columns=["time_to_eruption"])
    else:
        y = None
        X = df

    return X, y


def load_train_data(load_cached_data=True, debug=None):
    """
    Loads the training dataset.
    """
    return generate_dataset(
        Config.TRAIN_META_PATH,
        "train_features.parquet",
        load_cached_data=load_cached_data,
        debug=debug,
    )


def load_val_data(load_cached_data=True, debug=None):
    """
    Loads the validation dataset.
    """
    return generate_dataset(
        Config.VAL_META_PATH,
        "val_features.parquet",
        load_cached_data=load_cached_data,
        debug=debug,
    )


def load_test_data(load_cached_data=True, debug=None):
    """
    Loads the test dataset.
    """
    return generate_dataset(
        Config.TEST_META_PATH,
        "test_features.parquet",
        load_cached_data=load_cached_data,
        debug=debug,
    )
