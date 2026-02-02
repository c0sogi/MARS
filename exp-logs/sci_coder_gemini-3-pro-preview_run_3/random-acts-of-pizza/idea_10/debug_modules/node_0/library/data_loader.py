import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    TEXT_COL,
    TITLE_COL,
    SUBREDDIT_COL,
)
from library.utils import Timer, save_parquet, load_parquet


def clean_dataframe(df):
    """
    Performs basic cleaning on the dataframe.
    - Fills missing text fields with empty strings.
    - Ensures subreddit column is a list (handles NaNs).
    """
    # Avoid modifying the original dataframe
    df = df.copy()

    # Fill missing text columns with empty strings
    if TEXT_COL in df.columns:
        df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    if TITLE_COL in df.columns:
        df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)

    # Handle Subreddit Column
    # It is expected to be a list of strings.
    # If it's NaN or None, replace with empty list.
    if SUBREDDIT_COL in df.columns:
        # We use apply to safely handle mixed types (lists and NaNs)
        df[SUBREDDIT_COL] = df[SUBREDDIT_COL].apply(
            lambda x: x if isinstance(x, (list, np.ndarray)) else []
        )

    return df


def load_datasets(load_cached_data=True, debug_size=None):
    """
    Loads train, validation, and test datasets.
    Implements caching for the cleaned dataframes to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load cleaned data from the working directory.
        debug_size (int, optional): If provided, returns only the first N rows of each dataset.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths in the working directory
    train_cache = os.path.join(WORKING_DIR, "train_cleaned.parquet")
    val_cache = os.path.join(WORKING_DIR, "val_cleaned.parquet")
    test_cache = os.path.join(WORKING_DIR, "test_cleaned.parquet")

    with Timer("Load Datasets"):
        loaded_from_cache = False

        # 1. Try to load from cache
        if load_cached_data:
            if (
                os.path.exists(train_cache)
                and os.path.exists(val_cache)
                and os.path.exists(test_cache)
            ):
                print(f"Loading cleaned datasets from cache: {WORKING_DIR}")
                try:
                    train_df = load_parquet(train_cache)
                    val_df = load_parquet(val_cache)
                    test_df = load_parquet(test_cache)
                    loaded_from_cache = True
                except Exception as e:
                    print(f"Failed to load cache: {e}. Falling back to raw metadata.")
            else:
                print("Cache not found or incomplete. Loading from metadata...")

        # 2. Load from metadata if not loaded from cache
        if not loaded_from_cache:
            print(f"Loading raw metadata from: {os.path.dirname(TRAIN_PATH)}")
            train_df = load_parquet(TRAIN_PATH)
            val_df = load_parquet(VAL_PATH)
            test_df = load_parquet(TEST_PATH)

            # 3. Clean dataframes
            print("Cleaning dataframes...")
            train_df = clean_dataframe(train_df)
            val_df = clean_dataframe(val_df)
            test_df = clean_dataframe(test_df)

            # 4. Save to cache
            print(f"Saving cleaned datasets to cache: {WORKING_DIR}")
            save_parquet(train_df, train_cache)
            save_parquet(val_df, val_cache)
            save_parquet(test_df, test_cache)

    # 5. Handle Debugging (Subsampling)
    if debug_size is not None:
        print(f"Subsampling data to {debug_size} samples for debugging.")
        train_df = train_df.head(debug_size)
        val_df = val_df.head(debug_size)
        test_df = test_df.head(debug_size)

    return train_df, val_df, test_df
