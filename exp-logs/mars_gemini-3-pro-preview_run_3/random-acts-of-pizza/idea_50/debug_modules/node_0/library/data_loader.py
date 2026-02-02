import os
import pandas as pd
import numpy as np
from library.config import METADATA_DIR, WORKING_DIR, TEXT_COLS, SUBREDDIT_COL
from library.utils import Timer


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans text columns by filling NaNs with empty strings and enforcing string type.
    Also ensures the subreddit list column is properly formatted.
    """
    df = df.copy()

    # Clean text columns defined in config
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Clean subreddit column - ensure it's a list
    if SUBREDDIT_COL in df.columns:
        # If it's NaN, replace with empty list
        # Note: In parquet, lists are usually handled well, but we ensure safety here
        mask = df[SUBREDDIT_COL].isnull()
        if mask.any():
            df.loc[mask, SUBREDDIT_COL] = df.loc[mask, SUBREDDIT_COL].apply(
                lambda x: []
            )

    return df


def load_dataset(load_cached_data: bool = True):
    """
    Loads the train, validation, and test datasets.

    Implements a caching mechanism:
    1. Checks if cleaned parquet files exist in WORKING_DIR.
    2. If yes and load_cached_data is True, loads them.
    3. If no, loads raw metadata from METADATA_DIR, cleans them, saves to cache, and returns.

    Returns:
        tuple: (train_df, val_df, test_df)
    """

    # Define cache paths
    train_cache = os.path.join(WORKING_DIR, "train_cleaned.parquet")
    val_cache = os.path.join(WORKING_DIR, "val_cleaned.parquet")
    test_cache = os.path.join(WORKING_DIR, "test_cleaned.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        with Timer("Load Data from Cache"):
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df

    with Timer("Load and Clean Data from Metadata"):
        # Load raw metadata
        train_path = os.path.join(METADATA_DIR, "train.parquet")
        val_path = os.path.join(METADATA_DIR, "val.parquet")
        test_path = os.path.join(METADATA_DIR, "test.parquet")

        if not (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            raise FileNotFoundError(f"Metadata files not found in {METADATA_DIR}")

        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

        # Apply cleaning
        train_df = clean_text_fields(train_df)
        val_df = clean_text_fields(val_df)
        test_df = clean_text_fields(test_df)

        # Save to cache
        os.makedirs(WORKING_DIR, exist_ok=True)
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

        return train_df, val_df, test_df
