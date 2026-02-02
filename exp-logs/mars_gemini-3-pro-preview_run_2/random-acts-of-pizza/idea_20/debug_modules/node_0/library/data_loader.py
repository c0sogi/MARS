import os
import json
import pandas as pd
import numpy as np
from library.config import Config


def _load_raw_json(path):
    """Loads a JSON file from the specified path."""
    with open(path, "r") as f:
        return json.load(f)


def clean_text_fields(df):
    """
    Fills missing values in text columns with empty strings and ensures string type.
    """
    for col in Config.TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def format_subreddits(df):
    """
    Converts the list of subreddits into a space-separated string.
    """
    col = Config.SUBREDDIT_COL
    if col in df.columns:
        # Join list items with space; handle non-list values gracefully
        df[col] = df[col].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    return df


def _process_data_from_scratch(split, meta_path):
    """
    Loads metadata, fetches raw records, and applies preprocessing.
    """
    print(f"Processing {split} data from scratch...")

    # 1. Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    df_meta = pd.read_csv(meta_path)

    # 2. Identify Source Files
    # Metadata contains 'source_file' (e.g., 'input/train.json')
    # We load each unique source file once to minimize I/O
    source_files = df_meta["source_file"].unique()
    loaded_raw_data = {}

    for src in source_files:
        # Construct path relative to current working directory
        # Config.INPUT_DIR is './input', but source_file already includes 'input/'
        # We assume source_file is relative to the root directory.
        full_path = os.path.join(".", src)
        if full_path not in loaded_raw_data:
            if os.path.exists(full_path):
                loaded_raw_data[full_path] = _load_raw_json(full_path)
            else:
                raise FileNotFoundError(f"Raw data file not found: {full_path}")

    # 3. Extract Records
    # We use sample_index from metadata to grab the correct dict from the list
    records = []
    # Optimization: If all come from one file (common case)
    if len(source_files) == 1:
        src = source_files[0]
        full_path = os.path.join(".", src)
        raw_list = loaded_raw_data[full_path]

        # Use numpy indexing for speed if possible, otherwise list comprehension
        indices = df_meta["sample_index"].values
        records = [raw_list[i] for i in indices]
    else:
        # Mixed sources
        for _, row in df_meta.iterrows():
            src = row["source_file"]
            idx = row["sample_index"]
            full_path = os.path.join(".", src)
            records.append(loaded_raw_data[full_path][idx])

    df_raw = pd.DataFrame(records)

    # 4. Select and Clean Columns
    # Define columns to keep
    cols_to_keep = (
        Config.TEXT_COLS
        + [Config.SUBREDDIT_COL]
        + Config.NUMERICAL_COLS
        + ["request_id"]
    )

    # Filter columns that exist in raw data
    existing_cols = [c for c in cols_to_keep if c in df_raw.columns]
    df_processed = df_raw[existing_cols].copy()

    # 5. Merge Target Label
    # We rely on metadata for the ground truth label for train/val
    if Config.TARGET_COL in df_meta.columns:
        df_processed[Config.TARGET_COL] = df_meta[Config.TARGET_COL].values

    # 6. Apply Preprocessing
    df_processed = clean_text_fields(df_processed)
    df_processed = format_subreddits(df_processed)

    # 7. Ensure Numerical Types
    for col in Config.NUMERICAL_COLS:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(
                df_processed[col], errors="coerce"
            ).fillna(0)

    return df_processed


def load_dataset(split="train", load_cached_data=True, sample_size=None):
    """
    Main function to load dataset for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from parquet cache.
        sample_size (int, optional): If provided, returns a random subsample (for debugging).

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Determine paths based on split
    if split == "train":
        meta_path = Config.TRAIN_META_PATH
        cache_path = Config.TRAIN_FEATURES_PATH
    elif split == "val":
        meta_path = Config.VAL_META_PATH
        cache_path = Config.VAL_FEATURES_PATH
    elif split == "test":
        meta_path = Config.TEST_META_PATH
        cache_path = Config.TEST_FEATURES_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    df = None

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split} data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch if not loaded
    if df is None:
        df = _process_data_from_scratch(split, meta_path)

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"Saved processed {split} data to {cache_path}")

    # Subsampling for debugging
    if sample_size is not None and sample_size < len(df):
        print(f"Subsampling {split} data to {sample_size} rows.")
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)

    return df
