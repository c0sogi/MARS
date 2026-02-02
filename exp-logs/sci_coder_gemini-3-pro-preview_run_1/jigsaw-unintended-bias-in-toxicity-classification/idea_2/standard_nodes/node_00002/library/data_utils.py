import os
import pandas as pd
import numpy as np
from library.config import Config


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the text column and ensures correct data type.

    Args:
        df: Input DataFrame containing the text column.

    Returns:
        DataFrame with cleaned text column.
    """
    if Config.TEXT_COL in df.columns:
        # Fill NaN values with empty string
        df[Config.TEXT_COL] = df[Config.TEXT_COL].fillna("").astype(str)
    return df


def get_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts continuous toxicity target to binary label based on 0.5 threshold.

    Args:
        df: Input DataFrame containing the continuous target column.

    Returns:
        DataFrame with an additional binary_target column.
    """
    if Config.TARGET_COL in df.columns:
        df[Config.BINARY_TARGET_COL] = (df[Config.TARGET_COL] >= 0.5).astype(int)
    return df


def load_data(
    data_type: str, load_cached_data: bool = True, nrows: int = None
) -> pd.DataFrame:
    """
    Loads data (train, val, or test), performing preprocessing and caching.

    Args:
        data_type: One of 'train', 'val', 'test'.
        load_cached_data: Whether to try loading from the local parquet cache.
        nrows: Number of rows to return (useful for debugging/testing).

    Returns:
        Processed DataFrame.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define paths based on data_type
    if data_type == "train":
        raw_path = Config.TRAIN_DATA_PATH
        cache_path = os.path.join(Config.WORKING_DIR, "train_data.parquet")
    elif data_type == "val":
        raw_path = Config.VAL_DATA_PATH
        cache_path = os.path.join(Config.WORKING_DIR, "val_data.parquet")
    elif data_type == "test":
        raw_path = Config.TEST_DATA_PATH
        cache_path = os.path.join(Config.WORKING_DIR, "test_data.parquet")
    else:
        raise ValueError("data_type must be one of ['train', 'val', 'test']")

    df = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading {data_type} data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
        except Exception:
            # If load fails, we fall back to raw processing
            df = None

    # 2. If no cache or cache load failed, process from scratch
    if df is None:
        # print(f"Loading raw {data_type} data from: {raw_path}")
        df = pd.read_csv(raw_path)

        # Apply preprocessing
        df = preprocess_text(df)

        # Apply binary target generation for labeled sets
        if data_type in ["train", "val"]:
            df = get_binary_target(df)

        # Save to cache
        # print(f"Saving processed {data_type} data to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)

    # 3. Apply slicing if requested
    if nrows is not None:
        df = df.head(nrows)

    return df
