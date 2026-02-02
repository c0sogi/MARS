import os
import pandas as pd
from library.config import TRAIN_PATH, VAL_PATH, TEST_PATH, CACHE_DIR
from library.utils import Timer


def get_union_dataset(load_cached_data=True):
    """
    Loads the training and validation datasets from the metadata directory,
    merges them into a single 'Union Dataset', and returns it.

    Implements caching to store the merged dataframe in the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load the pre-merged dataset
                                 from the cache directory.

    Returns:
        pd.DataFrame: The combined training and validation data.
    """
    cache_path = os.path.join(CACHE_DIR, "union_dataset.parquet")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        with Timer("Loading Union Dataset from cache"):
            return pd.read_parquet(cache_path)

    with Timer("Creating Union Dataset from metadata"):
        # Load individual splits from metadata
        if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
            raise FileNotFoundError(
                f"Metadata files not found. Expected {TRAIN_PATH} and {VAL_PATH}"
            )

        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)

        # Merge datasets
        # ignore_index=True ensures a clean index for the new union dataset
        union_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        union_df.to_parquet(cache_path, index=False)

    return union_df


def get_test_dataset():
    """
    Loads the test dataset from the metadata directory.

    Returns:
        pd.DataFrame: The test data.
    """
    with Timer("Loading Test Dataset"):
        if not os.path.exists(TEST_PATH):
            raise FileNotFoundError(f"Metadata file not found: {TEST_PATH}")
        return pd.read_parquet(TEST_PATH)


def load_datasets():
    """
    Loads train, validation, and test datasets separately from metadata.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    with Timer("Loading all datasets separately"):
        if not os.path.exists(TRAIN_PATH):
            raise FileNotFoundError(f"Train metadata not found: {TRAIN_PATH}")
        if not os.path.exists(VAL_PATH):
            raise FileNotFoundError(f"Val metadata not found: {VAL_PATH}")
        if not os.path.exists(TEST_PATH):
            raise FileNotFoundError(f"Test metadata not found: {TEST_PATH}")

        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)
        test_df = pd.read_parquet(TEST_PATH)

    return train_df, val_df, test_df
