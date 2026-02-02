import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger(name="data_loader")


def _load_raw_json_as_df(json_path):
    """
    Loads a JSON file and converts it to a DataFrame indexed by request_id.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Raw data file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # Ensure request_id is unique and set as index for fast joining
    if "request_id" in df.columns:
        df = df.drop_duplicates(subset=["request_id"])
        df.set_index("request_id", inplace=True)
    return df


def _process_split(meta_df, raw_train_df, raw_test_df, split_name):
    """
    Merges metadata with raw data and extracts necessary features.
    """
    logger.info(f"Processing {split_name} split with {len(meta_df)} samples...")

    # Identify which raw source to use based on the 'source_file' column in metadata
    # However, since we have request_id, we can just join.
    # To be safe, we check where the IDs come from.
    # In this dataset, IDs are unique across train and test (mostly),
    # but we will look up in both to be robust or use the source_file hint.

    # We'll combine raw dfs for easier lookup since the dataset fits in memory
    # raw_train_df and raw_test_df indices are request_id

    # Initialize features list
    records = []

    # Numerical columns to extract
    num_cols = Config.NUMERICAL_COLS

    # Pre-fetch data to avoid DataFrame.loc overhead in a loop if possible,
    # but for ~2k-4k rows, loop or apply is fine.
    # Let's use a merge operation which is vectorized and faster.

    # 1. Merge Metadata with Raw Data
    # We need to know which raw DF to join with.
    # The metadata 'source_file' tells us: 'input/train.json' or 'input/test.json'

    # Split metadata by source
    meta_from_train = meta_df[meta_df["source_file"].str.contains("train.json")]
    meta_from_test = meta_df[meta_df["source_file"].str.contains("test.json")]

    merged_parts = []

    if not meta_from_train.empty:
        merged_train = meta_from_train.join(
            raw_train_df, on="request_id", how="left", rsuffix="_raw"
        )
        merged_parts.append(merged_train)

    if not meta_from_test.empty:
        merged_test = meta_from_test.join(
            raw_test_df, on="request_id", how="left", rsuffix="_raw"
        )
        merged_parts.append(merged_test)

    if not merged_parts:
        return pd.DataFrame()

    df_merged = pd.concat(merged_parts, axis=0)

    # Preserve original metadata order
    df_merged = meta_df[["request_id"]].merge(df_merged, on="request_id", how="left")

    # 2. Extract and Engineer Features

    # View 1: Semantic Text
    # Concatenate title and text. Handle NaNs.
    title = df_merged["request_title"].fillna("").astype(str)
    text = df_merged["request_text_edit_aware"].fillna("").astype(str)
    df_merged["full_text"] = title + " " + text

    # View 2: User Persona (Subreddits)
    # Convert list of subreddits to a space-separated string for easy TF-IDF vectorization later.
    # Handle cases where it might not be a list (e.g. NaN or float)
    def process_subreddits(x):
        if isinstance(x, list):
            return " ".join([str(s) for s in x])
        return ""

    df_merged["subreddit_text"] = df_merged[Config.SUBREDDIT_COL].apply(
        process_subreddits
    )

    # View 3: Robust Metadata
    # Ensure numerical columns are float
    for col in num_cols:
        if col not in df_merged.columns:
            logger.warning(
                f"Numerical column {col} missing in raw data. Filling with 0."
            )
            df_merged[col] = 0.0
        else:
            df_merged[col] = df_merged[col].fillna(0.0).astype(float)

    # Select final columns
    # We keep request_id, target (if exists), and the extracted feature columns
    keep_cols = ["request_id", "full_text", "subreddit_text"] + num_cols

    if "requester_received_pizza" in df_merged.columns:
        keep_cols.append("requester_received_pizza")

    df_final = df_merged[keep_cols].copy()

    return df_final


def load_data(load_cached_data=True):
    """
    Main function to load train, validation, and test datasets.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
                                 If False or cache missing, re-processes raw data.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            logger.info("Loading cached datasets from Parquet...")
            try:
                df_train = pd.read_parquet(train_cache)
                df_val = pd.read_parquet(val_cache)
                df_test = pd.read_parquet(test_cache)
                logger.info("Successfully loaded cached data.")
                return df_train, df_val, df_test
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-processing data.")
        else:
            logger.info("Cache not found. Processing raw data...")
    else:
        logger.info("Ignoring cache. Processing raw data...")

    # Load Metadata
    logger.info("Loading metadata...")
    meta_train = pd.read_csv(Config.TRAIN_META_PATH)
    meta_val = pd.read_csv(Config.VAL_META_PATH)
    meta_test = pd.read_csv(Config.TEST_META_PATH)

    # Load Raw JSONs
    logger.info("Loading raw JSON files...")
    raw_train = _load_raw_json_as_df(Config.TRAIN_JSON_PATH)
    raw_test = _load_raw_json_as_df(Config.TEST_JSON_PATH)

    # Process Splits
    df_train = _process_split(meta_train, raw_train, raw_test, "Train")
    df_val = _process_split(meta_val, raw_train, raw_test, "Validation")
    df_test = _process_split(meta_test, raw_train, raw_test, "Test")

    # Save to Cache
    logger.info("Saving processed datasets to cache...")
    df_train.to_parquet(train_cache, index=False)
    df_val.to_parquet(val_cache, index=False)
    df_test.to_parquet(test_cache, index=False)

    logger.info("Data loading and processing complete.")

    return df_train, df_val, df_test
