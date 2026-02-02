import os
import pandas as pd
import numpy as np
from library import config


def load_dataset(path, columns=None):
    """
    Loads a dataset from a Parquet file using the PyArrow engine for efficiency.

    Args:
        path (str): The file path to the Parquet dataset.
        columns (list, optional): List of column names to load. Defaults to None (all columns).

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at path: {path}")

    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def get_data_splits(load_cached_data=True):
    """
    Loads the full training dataset (for global knowledge aggregation), a stable
    training subsample (for model training), and the validation/test sets.

    Implements caching for the training subsample to ensure reproducibility.
    In DEBUG mode, caching is bypassed to ensure the subsample is a valid subset
    of the reduced global dataset.

    Args:
        load_cached_data (bool): If True, attempts to load the training subsample
                                 from the local cache.

    Returns:
        tuple: (full_train_df, train_subsample_df, val_df, test_df)
    """
    # Ensure working directory exists for caching
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    subsample_cache_path = os.path.join(config.WORKING_DIR, "train_subsample.parquet")

    # -------------------------------------------------------------------------
    # 1. Load Validation and Test Sets
    # -------------------------------------------------------------------------
    val_df = load_dataset(config.VAL_PATH)
    test_df = load_dataset(config.TEST_PATH)

    if config.DEBUG:
        # Slice validation set for rapid debugging
        val_df = val_df.iloc[: config.DEBUG_SIZE].copy()
        # Slice test set if necessary (though usually small enough)
        if len(test_df) > config.DEBUG_SIZE:
            test_df = test_df.iloc[: config.DEBUG_SIZE].copy()

    # -------------------------------------------------------------------------
    # 2. Load Full Training Set (Knowledge Base)
    # -------------------------------------------------------------------------
    # This dataset is used to compute global priors (Sum/Count stats)
    full_train_df = load_dataset(config.TRAIN_PATH)

    # -------------------------------------------------------------------------
    # 3. Generate or Load Training Subsample
    # -------------------------------------------------------------------------
    train_subsample_df = None

    if config.DEBUG:
        # In DEBUG mode, we must ensure consistency between the 'Global' set and
        # the 'Subsample'. We slice the global set and sample directly from it,
        # bypassing the cache to avoid mismatches.
        debug_full_size = min(len(full_train_df), config.DEBUG_SIZE * 5)
        full_train_df = full_train_df.iloc[:debug_full_size].copy()

        sample_size = min(len(full_train_df), config.DEBUG_SIZE)
        train_subsample_df = full_train_df.sample(
            n=sample_size, random_state=config.RANDOM_SEED
        ).copy()

    else:
        # Normal Mode: Attempt to load stable subsample from cache
        if load_cached_data and os.path.exists(subsample_cache_path):
            try:
                train_subsample_df = load_dataset(subsample_cache_path)
            except Exception:
                # Fallback to recomputing if cache load fails
                train_subsample_df = None

        # Compute if not loaded
        if train_subsample_df is None:
            if len(full_train_df) <= config.TRAIN_SUBSAMPLE_SIZE:
                train_subsample_df = full_train_df.copy()
            else:
                train_subsample_df = full_train_df.sample(
                    n=config.TRAIN_SUBSAMPLE_SIZE, random_state=config.RANDOM_SEED
                ).copy()

            # Save to cache for future runs
            train_subsample_df.to_parquet(subsample_cache_path, index=False)

    return full_train_df, train_subsample_df, val_df, test_df
