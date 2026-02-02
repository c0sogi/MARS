import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger, save_parquet, load_parquet

# Initialize Logger
logger = setup_logger("data_loader")


def load_raw_json(path):
    """
    Loads raw JSON data from the given path into a DataFrame.

    Args:
        path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the raw data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def load_dataset(split="train", load_cached_data=True):
    """
    Loads and processes the dataset for the specified split (train, val, test).

    This function handles:
    1. Checking for cached parquet files.
    2. Loading metadata and raw JSON data.
    3. Merging metadata with raw data to filter by split.
    4. Extracting text and numeric features.
    5. Caching the processed DataFrame.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed DataFrame with features and labels (if available).
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Resolve paths based on split
    if split == "train":
        meta_path = Config.TRAIN_META
        raw_path = Config.TRAIN_JSON
        cache_path = Config.TRAIN_FEATURES_PATH
    elif split == "val":
        meta_path = Config.VAL_META
        # Validation set is a subset of the training JSON file
        raw_path = Config.TRAIN_JSON
        cache_path = Config.VAL_FEATURES_PATH
    else:  # test
        meta_path = Config.TEST_META
        raw_path = Config.TEST_JSON
        cache_path = Config.TEST_FEATURES_PATH

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached {split} data from {cache_path}")
        return load_parquet(cache_path)

    logger.info(f"Processing {split} data from scratch...")

    # Load Metadata (defines the split and contains labels for train/val)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    df_meta = pd.read_csv(meta_path)

    # Load Raw JSON Data
    df_raw = load_raw_json(raw_path)

    # Merge Metadata with Raw Data
    # We drop the target from raw if it exists to rely on the metadata's target (for consistency)
    if Config.TARGET_COL in df_raw.columns and Config.TARGET_COL in df_meta.columns:
        df_raw = df_raw.drop(columns=[Config.TARGET_COL])

    # Left merge ensures we only keep records defined in the metadata split
    df_merged = df_meta.merge(df_raw, on=Config.ID_COL, how="left")

    # Feature Engineering: Text
    # Concatenate title and edit-aware text
    title = df_merged["request_title"].fillna("").astype(str)
    text = df_merged["request_text_edit_aware"].fillna("").astype(str)
    df_merged["text_concat"] = title + " " + text

    # Feature Engineering: Numerics
    # Ensure all required numeric columns exist and are filled
    for col in Config.NUMERIC_COLS:
        if col not in df_merged.columns:
            logger.warning(f"Numeric column {col} missing in raw data. Filling with 0.")
            df_merged[col] = 0.0
        else:
            df_merged[col] = df_merged[col].fillna(0.0)

    # Select Final Columns
    # ID + Text + Numerics + Target (if present)
    cols_to_keep = [Config.ID_COL, "text_concat"] + Config.NUMERIC_COLS

    if Config.TARGET_COL in df_merged.columns:
        cols_to_keep.append(Config.TARGET_COL)

    df_final = df_merged[cols_to_keep].copy()

    # Handle Debug Mode
    if Config.DEBUG:
        logger.info(f"DEBUG mode active: Sampling {Config.MAX_SAMPLES} rows.")
        df_final = df_final.head(Config.MAX_SAMPLES)

    # Save to Cache
    logger.info(f"Saving processed {split} data to {cache_path}")
    save_parquet(df_final, cache_path)

    return df_final
