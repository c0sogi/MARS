import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans text fields by filling missing values with empty strings.
    Uses Config.TEXT_COLS to identify text columns.

    Args:
        df (pd.DataFrame): The dataframe containing text columns.

    Returns:
        pd.DataFrame: Dataframe with cleaned text columns.
    """
    for key, col_name in Config.TEXT_COLS.items():
        if col_name in df.columns:
            # Fill NaN with empty string and ensure string type
            df[col_name] = df[col_name].fillna("").astype(str)
    return df


def get_feature_intersection(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list:
    """
    Identifies the intersection of columns between training and testing datasets
    to prevent feature leakage.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Testing dataframe.

    Returns:
        list: Sorted list of column names present in both dataframes.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Calculate intersection
    common_cols = list(train_cols.intersection(test_cols))

    return sorted(common_cols)


def load_raw_data(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads the raw data for a specific split (train, val, test).
    Implements caching using Parquet files to speed up subsequent runs.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded and basic-cleaned dataframe.
    """
    set_seed()

    # Construct cache filename
    cache_file = f"raw_{split}.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_file)

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # Load from source
    print(f"Loading raw {split} data from source...")
    if split == "train":
        source_path = Config.TRAIN_DATA_PATH
    elif split == "val":
        source_path = Config.VAL_DATA_PATH
    elif split == "test":
        source_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)

    # Apply basic text cleaning
    df = clean_text_fields(df)

    # Handle Debug Mode
    if Config.DEBUG:
        print(
            f"DEBUG mode active: Sampling {Config.DEBUG_SAMPLE_SIZE} rows for {split}."
        )
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
        print(f"Saved {split} data to cache at {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df
