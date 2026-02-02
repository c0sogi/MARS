import os
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    WORKING_DIR,
)
from library.features import process_dataset


def load_dataset(metadata_path, is_train=True, load_cached_data=True, debug_size=None):
    """
    Loads the dataset, processing raw files if necessary or loading from cache.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        is_train (bool): Flag indicating if this is training data.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug_size (int, optional): Number of files to process for debugging.

    Returns:
        X (pd.DataFrame): Feature matrix (contains segment_id).
        y (pd.Series or None): Target variable (time_to_eruption) if available, else None.
    """

    # 1. Determine the appropriate cache file path based on the metadata input
    # Use absolute paths to ensure reliable matching
    abs_meta_path = os.path.abspath(metadata_path)

    if abs_meta_path == os.path.abspath(TRAIN_META_PATH):
        cache_path = TRAIN_FEATURES_PATH
    elif abs_meta_path == os.path.abspath(VAL_META_PATH):
        cache_path = VAL_FEATURES_PATH
    elif abs_meta_path == os.path.abspath(TEST_META_PATH):
        cache_path = TEST_FEATURES_PATH
    else:
        # Fallback for any custom metadata files
        filename = os.path.basename(metadata_path)
        name_part = os.path.splitext(filename)[0]
        cache_path = os.path.join(WORKING_DIR, f"{name_part}_features.parquet")

    # 2. Caching Mechanism
    df = None

    # Ensure the working directory for cache exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            # Attempt to load from cache
            df = pd.read_parquet(cache_path)

            # If a debug size is specified, slice the cached data
            if debug_size is not None:
                df = df.head(debug_size)
        except Exception:
            # If loading fails (e.g., corrupt file), proceed to re-process
            df = None

    if df is None:
        # Process data from scratch using the library function
        df = process_dataset(metadata_path, INPUT_DIR, debug_size=debug_size)

        # Save the processed dataframe to cache
        df.to_parquet(cache_path, index=False)

    # 3. Split Features (X) and Target (y)
    if "time_to_eruption" in df.columns:
        y = df["time_to_eruption"]
        X = df.drop(columns=["time_to_eruption"])
    else:
        y = None
        X = df

    return X, y
