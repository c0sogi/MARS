import os
import pandas as pd
import numpy as np
from library.config import Config


def _clean_text(df):
    """
    Cleans and combines text columns.
    Uses 'request_text_edit_aware' and 'request_title'.
    Concatenates them into 'text_combined' for downstream vectorization.
    """
    # Ensure strings and handle NaNs
    title = df[Config.TEXT_TITLE_COL].fillna("").astype(str)
    body = df[Config.TEXT_BODY_COL].fillna("").astype(str)

    # Concatenate Title + Body
    df["text_combined"] = title + " " + body
    return df


def _process_history(df):
    """
    Converts the list of subreddits into a space-separated string
    suitable for TF-IDF vectorization (Bag of Communities).
    """

    def join_subreddits(x):
        if isinstance(x, list):
            return " ".join([str(s) for s in x])
        elif isinstance(x, np.ndarray):
            return " ".join([str(s) for s in x])
        return ""

    if Config.HISTORY_COL in df.columns:
        df["history_str"] = df[Config.HISTORY_COL].apply(join_subreddits)
    else:
        # Fallback if column missing (should not happen with correct metadata)
        df["history_str"] = ""
    return df


def _select_features(df, is_test=False):
    """
    Selects allow-listed features and target.
    Strictly removes potential leakage columns not in the allow-list.
    """
    # Start with ID and necessary processed columns
    cols_to_keep = [Config.ID_COL, "text_combined", "history_str"]

    # Add Allow-Listed Metadata columns
    # Ensure they exist in df before selecting
    existing_meta = [c for c in Config.METADATA_COLS if c in df.columns]
    cols_to_keep.extend(existing_meta)

    # Add Target if not test and present
    if not is_test and Config.TARGET_COL in df.columns:
        cols_to_keep.append(Config.TARGET_COL)

    return df[cols_to_keep].copy()


def _process_dataframe(df, is_test=False):
    """
    Applies all cleaning, processing, and feature selection steps.
    """
    # 1. Text Cleaning & Concatenation
    df = _clean_text(df)

    # 2. History Processing (List -> String)
    df = _process_history(df)

    # 3. Feature Selection (Leakage Prevention)
    df = _select_features(df, is_test=is_test)

    # 4. Basic Type Casting / Filling for Metadata
    # Ensures numerical stability for downstream models
    for col in Config.METADATA_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def load_dataset(load_cached_data=True):
    """
    Loads the dataset.

    Logic:
    1. If load_cached_data is True, tries to load processed parquets from cache.
    2. Otherwise, loads raw metadata parquets.
    3. Merges Train and Validation sets into a single Union Dataset.
    4. Processes features (Text, History, Metadata).
    5. Caches the result and returns.

    Returns:
        train_df (pd.DataFrame): Union of Train and Val, processed.
        test_df (pd.DataFrame): Processed Test data.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_union_processed.parquet")
    test_cache_path = os.path.join(cache_dir, "test_processed.parquet")

    # 1. Try loading from cache
    if load_cached_data:
        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            print(f"Loading cached data from {cache_dir}...")
            try:
                train_df = pd.read_parquet(train_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, test_df
            except Exception as e:
                print(f"Error loading cache: {e}. Reloading from source...")
        else:
            print("Cache not found or incomplete. Reloading from source...")

    # 2. Load raw metadata
    print("Loading raw metadata...")
    if not os.path.exists(Config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train data not found at {Config.TRAIN_DATA_PATH}")
    if not os.path.exists(Config.VAL_DATA_PATH):
        raise FileNotFoundError(f"Val data not found at {Config.VAL_DATA_PATH}")
    if not os.path.exists(Config.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {Config.TEST_DATA_PATH}")

    raw_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    raw_val = pd.read_parquet(Config.VAL_DATA_PATH)
    raw_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # 3. Union Strategy: Merge Train and Val
    print("Merging Train and Validation sets (Union Strategy)...")
    train_union = pd.concat([raw_train, raw_val], axis=0, ignore_index=True)

    # 4. Process Data
    print("Processing Train Union data...")
    train_df = _process_dataframe(train_union, is_test=False)

    print("Processing Test data...")
    test_df = _process_dataframe(raw_test, is_test=True)

    # 5. Save to Cache
    print(f"Caching processed data to {cache_dir}...")
    train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, test_df
