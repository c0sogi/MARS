import pandas as pd
import numpy as np
import os
import ast
import re
from library import config


def clean_text(text):
    """
    Performs basic text sanitization:
    - Handles NaN/None by returning empty string
    - Lowercases text
    - Replaces newlines/tabs with spaces
    - Removes excessive internal whitespace
    """
    if pd.isna(text) or text is None:
        return ""

    # Ensure string type
    text = str(text)

    # Lowercase
    text = text.lower()

    # Replace newlines and tabs with a single space
    text = re.sub(r"[\n\t\r]+", " ", text)

    # Collapse multiple spaces into one and strip leading/trailing
    text = re.sub(r"\s+", " ", text).strip()

    return text


def safe_parse_list(val):
    """
    Safely parses a string representation of a list into a Python list.
    Useful for reading list columns stored as strings in CSVs.
    """
    if isinstance(val, list):
        return val

    if pd.isna(val) or val == "" or val is None:
        return []

    try:
        # ast.literal_eval is safe for evaluating strings containing Python literals
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return parsed
        return []
    except (ValueError, SyntaxError):
        # Fallback: return empty list if parsing fails
        return []


def get_common_columns(train_df, test_df, exclude_cols=None):
    """
    Identifies the intersection of columns between train and test dataframes.
    Useful for ensuring the model only sees features present in both sets.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        exclude_cols (list, optional): List of columns to explicitly exclude.

    Returns:
        list: Sorted list of common column names.
    """
    if exclude_cols is None:
        exclude_cols = []

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Find intersection
    common = list(train_cols.intersection(test_cols))

    # Remove excluded columns
    final_cols = [c for c in common if c not in exclude_cols]

    return sorted(final_cols)


def load_data(load_cached_data=True):
    """
    Loads train, validation, and test datasets.

    Logic:
    1. Checks for cached Parquet files in config.WORKING_DIR.
    2. If found and load_cached_data is True, loads from cache.
    3. If not, reads raw CSVs from config.METADATA_DIR.
    4. Applies processing:
       - Parses 'requester_subreddits_at_request' from string to list.
       - Cleans text columns ('request_text_edit_aware', 'request_title').
       - Ensures target variable is boolean.
    5. Saves processed data to cache for future runs.
    6. Applies DEBUG_SAMPLE_SIZE if set in config.

    Returns:
        tuple: (df_train, df_val, df_test)
    """

    # Define cache file paths
    cache_train_path = os.path.join(config.WORKING_DIR, "train_processed.parquet")
    cache_val_path = os.path.join(config.WORKING_DIR, "val_processed.parquet")
    cache_test_path = os.path.join(config.WORKING_DIR, "test_processed.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    )

    if load_cached_data and cache_exists:
        print(f"Loading data from cache: {config.WORKING_DIR}")
        try:
            df_train = pd.read_parquet(cache_train_path)
            df_val = pd.read_parquet(cache_val_path)
            df_test = pd.read_parquet(cache_test_path)

            # Apply debug sampling if configured
            if config.DEBUG_SAMPLE_SIZE is not None:
                print(
                    f"Debug Mode: Sampling {config.DEBUG_SAMPLE_SIZE} rows from cache."
                )
                df_train = df_train.head(config.DEBUG_SAMPLE_SIZE)
                df_val = df_val.head(config.DEBUG_SAMPLE_SIZE)
                df_test = df_test.head(config.DEBUG_SAMPLE_SIZE)

            return df_train, df_val, df_test
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")
            # Fall through to raw loading logic

    print("Loading data from raw CSVs...")

    # Load raw CSVs
    df_train = pd.read_csv(config.TRAIN_PATH)
    df_val = pd.read_csv(config.VAL_PATH)
    df_test = pd.read_csv(config.TEST_PATH)

    # --- Data Processing ---

    # 1. Parse Subreddit Lists
    # The CSV format stores lists as strings like "['a', 'b']"
    print("Parsing subreddit lists...")
    list_col = "requester_subreddits_at_request"

    for df in [df_train, df_val, df_test]:
        if list_col in df.columns:
            df[list_col] = df[list_col].apply(safe_parse_list)

    # 2. Clean Text Fields
    print("Cleaning text fields...")
    # We prioritize the edit-aware text, but fallback to raw text if missing
    text_col = "request_text_edit_aware"
    fallback_col = "request_text"
    title_col = "request_title"

    for df in [df_train, df_val, df_test]:
        # Ensure main text column is populated
        if text_col not in df.columns and fallback_col in df.columns:
            df[text_col] = df[fallback_col]

        # Clean Body
        if text_col in df.columns:
            df[text_col] = df[text_col].fillna("")
            df[text_col] = df[text_col].apply(clean_text)

        # Clean Title
        if title_col in df.columns:
            df[title_col] = df[title_col].fillna("")
            df[title_col] = df[title_col].apply(clean_text)

    # 3. Ensure Target is Boolean
    target_col = "requester_received_pizza"
    if target_col in df_train.columns:
        df_train[target_col] = df_train[target_col].astype(bool)
    if target_col in df_val.columns:
        df_val[target_col] = df_val[target_col].astype(bool)

    # --- Caching ---
    print("Saving processed data to cache...")
    try:
        # Save full datasets before sampling
        df_train.to_parquet(cache_train_path, index=False)
        df_val.to_parquet(cache_val_path, index=False)
        df_test.to_parquet(cache_test_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    # --- Debug Sampling ---
    if config.DEBUG_SAMPLE_SIZE is not None:
        print(f"Debug Mode: Sampling {config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.head(config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(config.DEBUG_SAMPLE_SIZE)

    return df_train, df_val, df_test
