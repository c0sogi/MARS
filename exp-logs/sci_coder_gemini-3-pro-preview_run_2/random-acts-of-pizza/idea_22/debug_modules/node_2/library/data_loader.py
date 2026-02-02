import os
import json
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_JSON_PATH,
    TEST_JSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_DIR,
    NUMERIC_FEATURES,
    TEXT_COLS_TO_CONCAT,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import set_seed


def _load_raw_json_as_df(json_path):
    """
    Helper function to load a JSON file into a pandas DataFrame.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_labeled_data(load_cached_data=True, debug=DEBUG):
    """
    Loads the training and validation data, merges with metadata to identify splits,
    and returns a unified DataFrame with a 'split' column.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug (bool): If True, samples the dataset for debugging.

    Returns:
        pd.DataFrame: The unified labeled dataset.
    """
    cache_path = os.path.join(CACHE_DIR, "labeled_data.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading labeled data from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("Loading labeled data from source...")
        # 2. Load Metadata
        df_meta_train = pd.read_csv(TRAIN_META_PATH)
        df_meta_val = pd.read_csv(VAL_META_PATH)

        # Mark splits
        df_meta_train["split"] = "train"
        df_meta_val["split"] = "val"

        # 3. Load Raw Data
        df_raw = _load_raw_json_as_df(TRAIN_JSON_PATH)

        # 4. Merge
        # Drop target from raw if it exists to avoid conflicts/duplication with metadata target
        cols_to_drop = (
            ["requester_received_pizza"]
            if "requester_received_pizza" in df_raw.columns
            else []
        )
        df_raw_clean = df_raw.drop(columns=cols_to_drop, errors="ignore")

        # Merge metadata with raw data
        df_train = pd.merge(df_meta_train, df_raw_clean, on="request_id", how="left")
        df_val = pd.merge(df_meta_val, df_raw_clean, on="request_id", how="left")

        # Concatenate
        df = pd.concat([df_train, df_val], ignore_index=True)

        # Clean 'post_was_edited' to ensure homogeneous type for Parquet (Cite debug_lesson_1)
        if "post_was_edited" in df.columns:
            df["post_was_edited"] = (
                df["post_was_edited"].apply(lambda x: 1 if x else 0).astype(int)
            )

        # 5. Save to Cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    # Handle Debugging
    if debug:
        print(f"Debug mode: Sampling {DEBUG_SAMPLE_SIZE} rows from labeled data.")
        # Sample while maintaining some class balance or just random sample
        df = df.sample(n=min(len(df), DEBUG_SAMPLE_SIZE), random_state=42).reset_index(
            drop=True
        )

    return df


def load_test_data(load_cached_data=True, debug=DEBUG):
    """
    Loads the test data and merges with metadata.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug (bool): If True, samples the dataset for debugging.

    Returns:
        pd.DataFrame: The test dataset.
    """
    cache_path = os.path.join(CACHE_DIR, "test_data.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading test data from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("Loading test data from source...")
        # 2. Load Metadata
        df_meta_test = pd.read_csv(TEST_META_PATH)

        # 3. Load Raw Data
        df_raw = _load_raw_json_as_df(TEST_JSON_PATH)

        # 4. Merge
        # Test data does not have targets, so simple merge is fine
        df = pd.merge(df_meta_test, df_raw, on="request_id", how="left")

        # Clean 'post_was_edited' to ensure homogeneous type for Parquet (Cite debug_lesson_1)
        if "post_was_edited" in df.columns:
            df["post_was_edited"] = (
                df["post_was_edited"].apply(lambda x: 1 if x else 0).astype(int)
            )

        # 5. Save to Cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    # Handle Debugging
    if debug:
        print(f"Debug mode: Sampling {DEBUG_SAMPLE_SIZE} rows from test data.")
        df = df.sample(n=min(len(df), DEBUG_SAMPLE_SIZE), random_state=42).reset_index(
            drop=True
        )

    return df


def extract_text_data(df):
    """
    Extracts and concatenates text columns specified in config.

    Args:
        df (pd.DataFrame): The dataframe containing text columns.

    Returns:
        np.ndarray: Array of strings (concatenated text).
    """
    # Validate columns
    valid_cols = [c for c in TEXT_COLS_TO_CONCAT if c in df.columns]
    if not valid_cols:
        raise ValueError(
            f"None of the text columns {TEXT_COLS_TO_CONCAT} found in DataFrame."
        )

    # Concatenate
    # Start with the first column
    text_series = df[valid_cols[0]].fillna("").astype(str)

    # Append subsequent columns with a space separator
    for col in valid_cols[1:]:
        text_series = text_series + " " + df[col].fillna("").astype(str)

    return text_series.values


def extract_numeric_data(df):
    """
    Extracts numeric features specified in config.

    Args:
        df (pd.DataFrame): The dataframe containing numeric columns.

    Returns:
        np.ndarray: Array of floats (numeric features).
    """
    # Validate columns
    valid_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    # Warn if columns are missing
    if len(valid_cols) != len(NUMERIC_FEATURES):
        missing = set(NUMERIC_FEATURES) - set(valid_cols)
        print(
            f"Warning: The following numeric columns are missing and will be skipped: {missing}"
        )

    if not valid_cols:
        print("Warning: No valid numeric columns found. Returning empty array.")
        return np.zeros((len(df), 0))

    # Extract and fill NaNs with 0 (Standard imputation should happen in pipeline, this is a safety fill)
    X_num = df[valid_cols].fillna(0).values.astype(float)
    return X_num
