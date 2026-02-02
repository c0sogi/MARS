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

    # Modify cache path if debug_size is set to avoid overwriting full cache
    if debug_size is not None:
        base, ext = os.path.splitext(cache_path)
        cache_path = f"{base}_debug_{debug_size}{ext}"

    # 2. Caching Mechanism
    df = None

    # Ensure the working directory for cache exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Load metadata to verify expected count
    try:
        meta_df = pd.read_csv(metadata_path)
        expected_len = len(meta_df)
        if debug_size is not None:
            expected_len = min(expected_len, debug_size)
    except Exception as e:
        print(f"Warning: Could not read metadata for validation: {e}")
        expected_len = None

    if load_cached_data and os.path.exists(cache_path):
        try:
            # Attempt to load from cache
            print(f"Loading cached features from {cache_path}...")
            df = pd.read_parquet(cache_path)

            # Validation: Check if cache matches expected metadata length
            if expected_len is not None:
                if len(df) != expected_len:
                    print(
                        f"Cache validation failed! Found {len(df)} rows, expected {expected_len} rows."
                    )
                    print("Invalidating cache and reprocessing...")
                    df = None
                else:
                    print(f"Cache validated. Loaded {len(df)} rows.")

        except Exception as e:
            print(f"Failed to load cache: {e}")
            df = None

    if df is None:
        print(f"Processing dataset from {metadata_path}...")
        # Process data from scratch using the library function
        df = process_dataset(metadata_path, INPUT_DIR, debug_size=debug_size)

        # Save the processed dataframe to cache
        print(f"Saving features to {cache_path}...")
        df.to_parquet(cache_path, index=False)

    # 3. Split Features (X) and Target (y)
    if "time_to_eruption" in df.columns:
        y = df["time_to_eruption"]
        X = df.drop(columns=["time_to_eruption"])
    else:
        y = None
        X = df

    return X, y
