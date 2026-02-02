import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import save_parquet, load_parquet


def load_raw_json(path: str):
    """
    Loads a JSON file from the given path.
    """
    with open(path, "r") as f:
        return json.load(f)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and preprocesses text and numerical features from the raw DataFrame.
    """
    # Text Concatenation
    # Fill NaNs with empty string to ensure valid string concatenation
    title = df["request_title"].fillna("").astype(str)
    text = df["request_text_edit_aware"].fillna("").astype(str)

    # Concatenate title and text as per DBCE strategy
    df["text_combined"] = title + " " + text

    # Process Numerical Columns
    for col in Config.NUMERIC_COLS:
        if col in df.columns:
            # Force numeric type and fill NaNs with 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            # If column is missing in raw data (e.g. test set), fill with 0
            df[col] = 0.0

    # Select columns to keep
    cols_to_keep = [Config.ID_COL, "text_combined"] + Config.NUMERIC_COLS

    # Include target if present
    if Config.TARGET_COL in df.columns:
        cols_to_keep.append(Config.TARGET_COL)

    return df[cols_to_keep]


def process_split(metadata_path: str, raw_data_list: list) -> pd.DataFrame:
    """
    Creates a DataFrame for a specific split (train/val/test) using metadata indices.
    """
    # Load metadata to get indices and labels
    meta_df = pd.read_csv(metadata_path)

    # Use sample_index to extract specific records from the raw data list
    # This is much faster than merging on request_id
    indices = meta_df["sample_index"].values
    records = [raw_data_list[i] for i in indices]

    data_df = pd.DataFrame(records)

    # Ensure the target label comes from metadata (ground truth for splits)
    if Config.TARGET_COL in meta_df.columns:
        data_df[Config.TARGET_COL] = meta_df[Config.TARGET_COL].values

    # Extract relevant features
    processed_df = extract_features(data_df)

    return processed_df


def load_dataset(load_from_cache: bool = True):
    """
    Main function to load train, validation, and test datasets.
    Handles caching to avoid re-processing raw JSONs.

    Args:
        load_from_cache (bool): If True, tries to load from parquet cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    cache_dir = Config.CACHE_DIR
    train_cache_path = os.path.join(cache_dir, "train_data.parquet")
    val_cache_path = os.path.join(cache_dir, "val_data.parquet")
    test_cache_path = os.path.join(cache_dir, "test_data.parquet")

    # Attempt to load from cache
    if load_from_cache:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            print("Loading datasets from cache...")
            train_df = load_parquet(train_cache_path)
            val_df = load_parquet(val_cache_path)
            test_df = load_parquet(test_cache_path)
            return train_df, val_df, test_df
        else:
            print("Cache miss. Processing raw data...")

    # Load raw data
    print("Loading raw JSON files...")
    train_raw = load_raw_json(Config.TRAIN_JSON_PATH)
    test_raw = load_raw_json(Config.TEST_JSON_PATH)

    # Process splits
    print("Processing Train split...")
    train_df = process_split(Config.TRAIN_META_PATH, train_raw)

    print("Processing Validation split...")
    val_df = process_split(Config.VAL_META_PATH, train_raw)

    print("Processing Test split...")
    test_df = process_split(Config.TEST_META_PATH, test_raw)

    # Save to cache
    print(f"Saving processed datasets to {cache_dir}...")
    save_parquet(train_df, train_cache_path)
    save_parquet(val_df, val_cache_path)
    save_parquet(test_df, test_cache_path)

    return train_df, val_df, test_df
