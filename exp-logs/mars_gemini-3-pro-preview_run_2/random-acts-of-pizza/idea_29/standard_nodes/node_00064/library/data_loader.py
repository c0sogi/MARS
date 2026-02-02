import os
import json
import pandas as pd
import numpy as np
from library.utils import setup_logger, set_seed

# Define paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_29/"

# Define feature sets
# We select robust numerical metadata, explicitly including timestamp
# and excluding complex subreddit lists or sparse history details.
NUMERICAL_FEATURES = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    "unix_timestamp_of_request",
]


def load_json_data(file_path):
    """
    Loads raw JSON data from the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def process_raw_data(json_data, metadata_df, is_test=False):
    """
    Merges raw JSON data with metadata and extracts relevant features.

    Args:
        json_data (list): List of dictionaries from raw JSON.
        metadata_df (pd.DataFrame): Metadata defining the split.
        is_test (bool): Whether processing test data (no target).

    Returns:
        pd.DataFrame: Processed DataFrame with features and target (if applicable).
    """
    # Convert raw list to DataFrame
    df_raw = pd.DataFrame(json_data)

    # Merge with metadata to filter rows and align with the split
    # metadata_df contains 'request_id' and for train/val 'requester_received_pizza'
    # Cite debug_lesson_9: Handle Column Overlaps Explicitly in Pandas Joins
    df_merged = pd.merge(
        metadata_df, df_raw, on="request_id", how="left", suffixes=("", "_raw")
    )

    # --- Text Extraction ---
    # Concatenate title and edit-aware text for a comprehensive semantic view
    df_merged["request_title"] = df_merged["request_title"].fillna("")
    df_merged["request_text_edit_aware"] = df_merged["request_text_edit_aware"].fillna(
        ""
    )

    df_merged["text_combined"] = (
        df_merged["request_title"] + " " + df_merged["request_text_edit_aware"]
    )

    # --- Numerical Feature Extraction ---
    # Ensure all expected numerical columns exist and are filled
    for col in NUMERICAL_FEATURES:
        if col not in df_merged.columns:
            df_merged[col] = 0.0
        else:
            df_merged[col] = df_merged[col].fillna(0.0)

    # Define columns to return
    cols_to_keep = ["request_id", "text_combined"] + NUMERICAL_FEATURES

    # Add target if available and not in test mode
    if not is_test:
        if "requester_received_pizza" in df_merged.columns:
            cols_to_keep.append("requester_received_pizza")

    return df_merged[cols_to_keep].copy()


def get_data_splits(load_cached_data=True):
    """
    Main entry point to retrieve Train, Validation, and Test DataFrames.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    logger = setup_logger("data_loader", os.path.join(WORKING_DIR, "data_loader.log"))

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_cache_path = os.path.join(WORKING_DIR, "train_processed.parquet")
    val_cache_path = os.path.join(WORKING_DIR, "val_processed.parquet")
    test_cache_path = os.path.join(WORKING_DIR, "test_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            logger.info("Loading processed data from cache...")
            try:
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)

                # Validate cache integrity: Cite debug_lesson_4
                if "requester_received_pizza" not in train_df.columns:
                    raise ValueError(
                        "Cached training data is missing target column 'requester_received_pizza'"
                    )

                return train_df, val_df, test_df
            except Exception as e:
                logger.warning(f"Cache load failed ({e}). Reprocessing from scratch...")
        else:
            logger.info("Cache files not found. Processing from scratch...")
    else:
        logger.info("Forced reload. Processing from scratch...")

    # 2. Load Raw Data
    logger.info("Reading raw JSON files...")
    train_json = load_json_data(os.path.join(INPUT_DIR, "train.json"))
    test_json = load_json_data(os.path.join(INPUT_DIR, "test.json"))

    # 3. Load Metadata
    logger.info("Reading metadata splits...")
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 4. Process Splits
    # Note: train.json contains the pool for both train and val splits.
    # The metadata DataFrame filters this pool to the specific rows for each split.
    logger.info("Processing Training Set...")
    train_df = process_raw_data(train_json, train_meta, is_test=False)

    logger.info("Processing Validation Set...")
    val_df = process_raw_data(train_json, val_meta, is_test=False)

    logger.info("Processing Test Set...")
    test_df = process_raw_data(test_json, test_meta, is_test=True)

    # 5. Save to Cache
    logger.info(f"Saving processed data to {WORKING_DIR}...")
    train_df.to_parquet(train_cache_path, index=False)
    val_df.to_parquet(val_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    logger.info("Data loading and processing complete.")

    return train_df, val_df, test_df
