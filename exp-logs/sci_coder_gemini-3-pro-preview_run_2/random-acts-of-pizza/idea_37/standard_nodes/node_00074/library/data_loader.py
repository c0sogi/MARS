import os
import json
import pandas as pd
import numpy as np
from library.config import Config


def preprocess_text(row):
    """
    Concatenates title and edit-aware text.
    """
    parts = []
    for col in Config.TEXT_COLS:
        val = row.get(col)
        if val is not None and isinstance(val, str):
            parts.append(val)
        elif val is not None:
            parts.append(str(val))
        else:
            parts.append("")
    return " ".join(parts).strip()


def process_split(meta_df, raw_data, is_test=False):
    """
    Extracts features from raw_data based on indices in meta_df.
    """
    records = []

    # Optimize lookup by converting raw_data list to direct access if not already
    # raw_data is expected to be a list of dicts

    for _, meta_row in meta_df.iterrows():
        idx = int(meta_row["sample_index"])
        raw_entry = raw_data[idx]

        record = {}
        record["request_id"] = meta_row["request_id"]

        # Text Processing
        record["text_combined"] = preprocess_text(raw_entry)

        # Numerical Features
        for col in Config.NUMERICAL_COLS:
            # Handle potential missing values safely
            val = raw_entry.get(col, np.nan)
            record[col] = val

        # Target Variable
        if not is_test:
            # We trust the metadata label as the ground truth for the split
            record[Config.TARGET_COL] = int(meta_row[Config.TARGET_COL])

        records.append(record)

    return pd.DataFrame(records)


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, merging metadata with raw JSON data.
    Implements caching using Parquet files.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")

    # Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading datasets from cache...")
            df_train = pd.read_parquet(train_cache)
            df_val = pd.read_parquet(val_cache)
            df_test = pd.read_parquet(test_cache)
            return df_train, df_val, df_test
        else:
            print("Cache not found. Processing from scratch...")
    else:
        print("Ignoring cache. Processing from scratch...")

    # Load Metadata
    print("Loading metadata...")
    df_meta_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_meta_val = pd.read_csv(Config.VAL_META_PATH)
    df_meta_test = pd.read_csv(Config.TEST_META_PATH)

    # Load Raw Data
    print("Loading raw JSON data...")
    with open(Config.TRAIN_JSON, "r") as f:
        raw_train_data = json.load(f)

    with open(Config.TEST_JSON, "r") as f:
        raw_test_data = json.load(f)

    # Process Splits
    print("Processing training set...")
    df_train = process_split(df_meta_train, raw_train_data, is_test=False)

    print("Processing validation set...")
    df_val = process_split(df_meta_val, raw_train_data, is_test=False)

    print("Processing test set...")
    df_test = process_split(df_meta_test, raw_test_data, is_test=True)

    # Save to Cache
    print("Saving processed datasets to cache...")
    df_train.to_parquet(train_cache, index=False)
    df_val.to_parquet(val_cache, index=False)
    df_test.to_parquet(test_cache, index=False)

    return df_train, df_val, df_test
