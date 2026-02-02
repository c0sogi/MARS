import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    RETRIEVAL_SUFFIX,
    N_FOLDS,
    SEED,
)
from library.utils import set_seed


def load_dataset(load_cached_data=True):
    """
    Loads the train, validation, and test datasets.

    Implements caching to store cleaned versions of the data (leakage columns removed).

    Args:
        load_cached_data (bool): If True, attempts to load processed files from the cache directory.
                                 If False or if files are missing, re-processes raw data.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    set_seed(SEED)

    # Define cache paths
    train_cache = os.path.join(CACHE_DIR, "train_base.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_base.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_base.parquet")

    # Check if cached files exist
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached datasets from {CACHE_DIR}...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        return train_df, val_df, test_df

    print("Loading raw datasets from metadata...")
    # Load raw data from metadata directory
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Identify leakage columns (stats at retrieval time)
    # We only look at train_df columns because test_df shouldn't have them
    leakage_cols = [c for c in train_df.columns if c.endswith(RETRIEVAL_SUFFIX)]

    if leakage_cols:
        print(
            f"Dropping {len(leakage_cols)} leakage columns ending with '{RETRIEVAL_SUFFIX}'..."
        )
        train_df = train_df.drop(columns=leakage_cols)
        val_df = val_df.drop(columns=leakage_cols)

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Save to cache
    print(f"Saving processed datasets to {CACHE_DIR}...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def get_stratified_cv(n_splits=N_FOLDS, shuffle=True, random_state=SEED):
    """
    Creates a StratifiedKFold cross-validator.

    Args:
        n_splits (int): Number of folds.
        shuffle (bool): Whether to shuffle each class's samples before splitting.
        random_state (int): Random seed for reproducibility.

    Returns:
        StratifiedKFold: The cross-validator object.
    """
    return StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
