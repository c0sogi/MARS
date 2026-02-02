import os
import ast
import re
import pandas as pd
import numpy as np
from library.config import Config


def clean_text(text):
    """
    Performs basic string normalization on text data.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text (lowercased, whitespace cleaned).
    """
    if pd.isna(text):
        return ""

    # Convert to string if not already
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove excessive whitespace (newlines, tabs, multiple spaces)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_common_columns(train_df, test_df):
    """
    Identifies the intersection of columns between training and test datasets.
    This helps in selecting only those features that are available at inference time,
    preventing data leakage (e.g., columns present in train but not test).

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.

    Returns:
        list: List of column names present in both dataframes.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Intersection
    common_cols = list(train_cols.intersection(test_cols))

    # Sort for deterministic order
    common_cols.sort()

    return common_cols


def load_dataset(load_cached_data=True):
    """
    Loads the Train, Validation, and Test datasets.

    Implements a caching mechanism using Parquet files to speed up subsequent executions.
    Parses stringified list columns (e.g., subreddit history) into actual Python lists.
    Applies basic text cleaning to text columns.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.
                                 If False or cache miss, loads from raw metadata CSVs, processes, and caches.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache_path = os.path.join(Config.WORKING_DIR, "train_base.parquet")
    val_cache_path = os.path.join(Config.WORKING_DIR, "val_base.parquet")
    test_cache_path = os.path.join(Config.WORKING_DIR, "test_base.parquet")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            try:
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, val_df, test_df
            except Exception:
                # Fallback to reloading if cache is corrupt
                pass

    # Load from metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # --- Preprocessing ---

    # 1. Parse History Column (String -> List)
    # The subreddit history is stored as a string representation of a list in the CSVs.
    history_col = Config.HISTORY_COL

    def parse_list(val):
        if isinstance(val, str):
            try:
                # Safely evaluate string literal to list
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
        elif isinstance(val, list):
            return val
        return []

    for df in [train_df, val_df, test_df]:
        if history_col in df.columns:
            df[history_col] = df[history_col].apply(parse_list)

    # 2. Clean Text Columns
    text_cols = Config.TEXT_COLS
    for df in [train_df, val_df, test_df]:
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

    # --- Caching ---
    try:
        # Save to parquet. Pyarrow handles list columns automatically.
        train_df.to_parquet(train_cache_path, index=False)
        val_df.to_parquet(val_cache_path, index=False)
        test_df.to_parquet(test_cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save to cache: {e}")

    return train_df, val_df, test_df
