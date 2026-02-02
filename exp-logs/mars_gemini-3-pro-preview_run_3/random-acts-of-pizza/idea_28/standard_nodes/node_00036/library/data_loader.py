import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import Timer


def clean_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic cleaning on text and list columns to ensure type safety.
    - Fills NaNs in text columns with empty strings.
    - Ensures the subreddit list column contains lists (handling NaNs).
    """
    df = df.copy()

    # Clean Text Columns
    text_cols = [Config.TEXT_COL, Config.TITLE_COL]
    for col in text_cols:
        if col in df.columns:
            # Fill NaN with empty string and ensure string type
            df[col] = df[col].fillna("").astype(str)

    # Clean Subreddit List Column
    # Ensure every entry is a list, even if it was NaN or None
    if Config.SUBREDDIT_COL in df.columns:

        def ensure_list(x):
            if isinstance(x, (list, np.ndarray)):
                return list(x)
            return []

        df[Config.SUBREDDIT_COL] = df[Config.SUBREDDIT_COL].apply(ensure_list)

    return df


def load_datasets(load_cached_data: bool = True):
    """
    Loads the train, validation, and test datasets.
    Implements a caching mechanism to store and retrieve cleaned dataframes.

    Args:
        load_cached_data (bool): If True, attempts to load from the cache directory first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure the working directory for caching exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define paths for cached cleaned files
    train_cache_path = os.path.join(Config.CACHE_DIR, "train_cleaned.parquet")
    val_cache_path = os.path.join(Config.CACHE_DIR, "val_cleaned.parquet")
    test_cache_path = os.path.join(Config.CACHE_DIR, "test_cleaned.parquet")

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    )

    # Attempt to load from cache
    if load_cached_data and cache_exists:
        with Timer("Load Datasets (Cached)"):
            train_df = pd.read_parquet(train_cache_path)
            val_df = pd.read_parquet(val_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            print(f"Successfully loaded cleaned datasets from {Config.CACHE_DIR}")
            return train_df, val_df, test_df

    # Fallback: Load raw metadata, clean, and cache
    with Timer("Load Datasets (Raw & Clean)"):
        # Verify raw files exist
        if not os.path.exists(Config.TRAIN_PATH):
            raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_PATH}")

        # Load raw data
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        val_df = pd.read_parquet(Config.VAL_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)

        # Apply cleaning logic
        train_df = clean_text_data(train_df)
        val_df = clean_text_data(val_df)
        test_df = clean_text_data(test_df)

        # Save cleaned data to cache
        train_df.to_parquet(train_cache_path, index=False)
        val_df.to_parquet(val_cache_path, index=False)
        test_df.to_parquet(test_cache_path, index=False)

        print(f"Processed raw data and saved to cache at {Config.CACHE_DIR}")

    return train_df, val_df, test_df
