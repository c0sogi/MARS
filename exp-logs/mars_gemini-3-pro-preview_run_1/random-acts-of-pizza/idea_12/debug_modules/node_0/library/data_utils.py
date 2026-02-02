import os
import ast
import pandas as pd
import numpy as np
from library.config import Config


def clean_text(text_series):
    """
    Cleans text columns by filling NaNs with empty strings and ensuring string type.

    Args:
        text_series (pd.Series): Series containing text data.

    Returns:
        pd.Series: Cleaned text series.
    """
    return text_series.fillna("").astype(str)


def parse_list_column(series):
    """
    Parses stringified lists (from CSV) back to list objects.
    Handles malformed strings or NaNs gracefully.

    Args:
        series (pd.Series): Series containing stringified lists.

    Returns:
        pd.Series: Series containing Python lists.
    """

    def safe_eval(x):
        try:
            if pd.isna(x) or x == "":
                return []
            # If it's already a list (e.g. loaded from parquet), return it
            if isinstance(x, list):
                return x
            # If it's a string representation of a list
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    return series.apply(safe_eval)


def get_common_columns(train_df, test_df, exclude_cols=None):
    """
    Identifies common columns between train and test dataframes,
    excluding specified columns (like targets or IDs).

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        exclude_cols (list, optional): List of columns to exclude.

    Returns:
        list: List of common column names.
    """
    if exclude_cols is None:
        exclude_cols = []

    common_cols = [
        col
        for col in train_df.columns
        if col in test_df.columns and col not in exclude_cols
    ]
    return common_cols


def load_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Loads, preprocesses, and caches the dataset.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): Whether to run in debug mode (sample data).

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache filenames based on debug status to avoid overwriting full cache with samples
    suffix = "_debug" if debug else ""
    train_cache = os.path.join(Config.WORKING_DIR, f"train_cleaned{suffix}.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, f"val_cleaned{suffix}.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, f"test_cleaned{suffix}.parquet")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print(f"Loading cached data from {Config.WORKING_DIR} (Debug={debug})...")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Error loading cache: {e}. Reloading from source.")
        else:
            print("Cache not found. Processing from scratch...")

    # Load raw data from Metadata CSVs
    print("Loading data from metadata CSVs...")
    if not os.path.exists(Config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train data not found at {Config.TRAIN_DATA_PATH}")

    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Apply Debug Sampling if enabled
    if debug:
        print(f"Debug mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows...")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Preprocessing
    print("Preprocessing text and list columns...")

    text_cols = ["request_text", "request_title", "request_text_edit_aware"]
    # requester_subreddits_at_request is stored as a stringified list in CSV
    list_cols = ["requester_subreddits_at_request"]

    for df in [train_df, val_df, test_df]:
        # Clean text columns
        for col in text_cols:
            if col in df.columns:
                df[col] = clean_text(df[col])

        # Parse list columns
        for col in list_cols:
            if col in df.columns:
                df[col] = parse_list_column(df[col])

    # Cache the processed data
    print(f"Caching processed data to {Config.WORKING_DIR}...")
    try:
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)
    except Exception as e:
        print(f"Warning: Could not cache data: {e}")

    return train_df, val_df, test_df
