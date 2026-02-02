import os
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    FEATURE_COLS,
    TARGET_COL,
    ID_COL,
    NUMERIC_TYPE,
)


def load_dataset(split="train", load_cached_data=True):
    """
    Loads the dataset for the specified split with caching support.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load pre-processed data from the working directory.

    Returns:
        tuple: (X, y, ids)
            X (pd.DataFrame): Feature matrix with float64 precision.
            y (np.ndarray or None): Target vector (species names) for train/val, None for test.
            ids (np.ndarray): Array of image identifiers.
    """
    # Ensure the working directory exists for caching
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache file paths
    # We use parquet for X to preserve column names and types reliably
    # We use npy for y and ids as they are simple 1D arrays
    cache_X_path = os.path.join(WORKING_DIR, f"X_{split}.parquet")
    cache_y_path = os.path.join(WORKING_DIR, f"y_{split}.npy")
    cache_ids_path = os.path.join(WORKING_DIR, f"ids_{split}.npy")

    # Attempt to load from cache
    if load_cached_data:
        # Check if essential files exist
        has_X = os.path.exists(cache_X_path)
        has_ids = os.path.exists(cache_ids_path)

        # For train/val, we also need y
        has_y = os.path.exists(cache_y_path)

        if split == "test" and has_X and has_ids:
            print(f"Loading cached {split} data from {WORKING_DIR}...")
            X = pd.read_parquet(cache_X_path)
            ids = np.load(cache_ids_path, allow_pickle=True)
            return X, None, ids

        elif split in ["train", "val"] and has_X and has_ids and has_y:
            print(f"Loading cached {split} data from {WORKING_DIR}...")
            X = pd.read_parquet(cache_X_path)
            y = np.load(cache_y_path, allow_pickle=True)
            ids = np.load(cache_ids_path, allow_pickle=True)
            return X, y, ids

    # If cache miss or reload forced, process from metadata
    print(f"Processing {split} data from metadata...")

    # Determine source path
    if split == "train":
        source_path = TRAIN_METADATA_PATH
    elif split == "val":
        source_path = VAL_METADATA_PATH
    elif split == "test":
        source_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found at {source_path}")

    # Load raw metadata
    df = pd.read_csv(source_path)

    # Extract and validate features
    # We strictly enforce the column order defined in FEATURE_COLS
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing feature columns in {split} set: {missing_features[:5]}..."
        )

    # Cast to high-precision float64 as required by the strategy
    X = df[FEATURE_COLS].astype(NUMERIC_TYPE)

    # Extract IDs
    ids = df[ID_COL].values

    # Extract Target if available
    y = None
    if split in ["train", "val"]:
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in {split} set.")
        y = df[TARGET_COL].values

    # Save to cache
    print(f"Saving processed {split} data to cache at {WORKING_DIR}...")
    X.to_parquet(cache_X_path, index=False)
    np.save(cache_ids_path, ids)

    if y is not None:
        np.save(cache_y_path, y)

    return X, y, ids
