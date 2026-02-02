import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger

# Initialize Logger
logger = setup_logger("data_loader")

# Define the numerical features to be extracted based on the "Robust Metadata" view
NUMERIC_COLS = [
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


def extract_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenates request title and body into a single text column.
    Handles missing values by filling them with empty strings.
    """
    logger.info("Extracting and combining text fields...")

    # Ensure columns exist and fill NaNs
    title = df.get("request_title", pd.Series([""] * len(df))).fillna("").astype(str)

    # Use edit_aware text if available, else fallback to request_text, else empty
    if "request_text_edit_aware" in df.columns:
        body = df["request_text_edit_aware"].fillna("").astype(str)
    elif "request_text" in df.columns:
        body = df["request_text"].fillna("").astype(str)
    else:
        body = pd.Series([""] * len(df)).astype(str)

    # Combine: Title + " " + Body
    df["combined_text"] = title + " " + body
    return df


def extract_subreddit_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the list of subreddits into a space-separated string.
    This prepares the data for CountVectorizer used in the LDA pipeline.
    """
    logger.info("Formatting subreddit lists...")

    def process_subreddit_list(sub_list):
        if isinstance(sub_list, list):
            # Join with spaces to treat as a 'document' of words
            return " ".join([str(s) for s in sub_list])
        return ""

    if "requester_subreddits_at_request" in df.columns:
        df["subreddit_list_str"] = df["requester_subreddits_at_request"].apply(
            process_subreddit_list
        )
    else:
        logger.warning(
            "'requester_subreddits_at_request' not found. Filling with empty strings."
        )
        df["subreddit_list_str"] = ""

    return df


def extract_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures all required numerical columns are present and cast to float.
    Missing columns are filled with 0.
    """
    logger.info("Extracting numerical features...")

    for col in NUMERIC_COLS:
        if col not in df.columns:
            logger.warning(f"Numerical column {col} missing. Filling with 0.")
            df[col] = 0.0
        else:
            # Coerce to float, fill NaNs with 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def _load_and_process_split(
    json_path: str, meta_path: str, output_path: str, is_test: bool = False
) -> pd.DataFrame:
    """
    Internal helper to load raw JSON, merge with metadata, process features,
    and save to Parquet.
    """
    logger.info(f"Processing split from {meta_path}...")

    # 1. Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    df_meta = pd.read_csv(meta_path)

    # 2. Load Raw JSON
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Raw JSON file not found: {json_path}")
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    df_raw = pd.DataFrame(raw_data)

    # 3. Merge
    # We merge on request_id. Metadata determines the split membership.
    # We use inner join implicitly by merging left on meta, but practically
    # every meta ID should exist in raw.
    df_merged = df_meta.merge(
        df_raw, on="request_id", how="left", suffixes=("", "_raw")
    )

    # Handle potential duplicate columns from merge (keep metadata version if conflict, though usually safe)
    if "requester_received_pizza_raw" in df_merged.columns:
        df_merged.drop(columns=["requester_received_pizza_raw"], inplace=True)

    # 4. Debug Subsampling
    if Config.DEBUG:
        logger.info(
            f"DEBUG mode enabled. Subsampling to {Config.DEBUG_SAMPLE_SIZE} rows."
        )
        df_merged = df_merged.head(Config.DEBUG_SAMPLE_SIZE).copy()

    # 5. Feature Extraction
    df_merged = extract_text_fields(df_merged)
    df_merged = extract_subreddit_lists(df_merged)
    df_merged = extract_numerical_features(df_merged)

    # 6. Select Final Columns
    # We keep IDs, target (if not test), and processed features
    cols_to_keep = ["request_id", "combined_text", "subreddit_list_str"] + NUMERIC_COLS

    if not is_test:
        if "requester_received_pizza" not in df_merged.columns:
            raise KeyError(
                "Target 'requester_received_pizza' missing in training/validation data."
            )
        cols_to_keep.append("requester_received_pizza")

    # Ensure we only keep columns that exist (robustness)
    cols_to_keep = [c for c in cols_to_keep if c in df_merged.columns]
    df_final = df_merged[cols_to_keep].copy()

    # 7. Save to Cache
    Config.ensure_directories()
    logger.info(f"Saving processed data to {output_path}")
    df_final.to_parquet(output_path, index=False)

    return df_final


def load_dataset(load_cached_data: bool = True):
    """
    Main entry point to load Train, Validation, and Test datasets.

    Args:
        load_cached_data (bool): If True, attempts to load from Parquet cache.
                                 If False or cache missing, re-processes from raw JSON.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    Config.ensure_directories()

    # Define tasks: (Name, JSON Path, Meta Path, Output Path, Is Test)
    tasks = [
        (
            "Train",
            Config.TRAIN_JSON_PATH,
            Config.TRAIN_META_PATH,
            Config.TRAIN_FEATURES_PATH,
            False,
        ),
        (
            "Val",
            Config.TRAIN_JSON_PATH,
            Config.VAL_META_PATH,
            Config.VAL_FEATURES_PATH,
            False,
        ),
        (
            "Test",
            Config.TEST_JSON_PATH,
            Config.TEST_META_PATH,
            Config.TEST_FEATURES_PATH,
            True,
        ),
    ]

    results = []

    for name, json_p, meta_p, out_p, is_test in tasks:
        # Check Cache
        if load_cached_data and os.path.exists(out_p):
            logger.info(f"Loading cached {name} data from {out_p}")
            df = pd.read_parquet(out_p)

            # Verify Debug constraint: if we are in debug mode but loaded a full dataset, slice it
            if Config.DEBUG and len(df) > Config.DEBUG_SAMPLE_SIZE:
                logger.info(
                    f"Slicing cached {name} data to {Config.DEBUG_SAMPLE_SIZE} for DEBUG mode."
                )
                df = df.head(Config.DEBUG_SAMPLE_SIZE)
        else:
            logger.info(
                f"Cache miss or reload requested for {name}. Processing from scratch."
            )
            df = _load_and_process_split(json_p, meta_p, out_p, is_test)

        results.append(df)

    return results[0], results[1], results[2]
