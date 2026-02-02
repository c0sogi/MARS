import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("DataManager")


def load_dataset(load_cached_data: bool = Config.LOAD_CACHED_DATA):
    """
    Loads the training, validation, and test datasets.

    If cached parquet files exist and load_cached_data is True, loads from disk.
    Otherwise, loads raw JSON and metadata, processes features, saves to cache, and returns.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    cache_dir = Config.WORKING_DIR
    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            logger.info("Loading datasets from cache...")
            try:
                df_train = pd.read_parquet(train_cache)
                df_val = pd.read_parquet(val_cache)
                df_test = pd.read_parquet(test_cache)

                # Apply Debug Slicing if needed
                if Config.DEBUG:
                    logger.info(
                        f"Debug mode enabled. Slicing datasets to {Config.DEBUG_SIZE} samples."
                    )
                    df_train = df_train.head(Config.DEBUG_SIZE)
                    df_val = df_val.head(Config.DEBUG_SIZE)
                    df_test = df_test.head(Config.DEBUG_SIZE)

                return df_train, df_val, df_test
            except Exception as e:
                logger.warning(
                    f"Failed to load cache: {e}. Re-processing from scratch."
                )
        else:
            logger.info("Cache files not found. Processing from scratch...")
    else:
        logger.info("Cache loading disabled. Processing from scratch...")

    # 2. Load Raw Data and Metadata
    logger.info("Loading raw JSON files...")
    with open(Config.TRAIN_JSON, "r") as f:
        raw_train_data = json.load(f)

    with open(Config.TEST_JSON, "r") as f:
        raw_test_data = json.load(f)

    logger.info("Loading metadata CSVs...")
    meta_train = pd.read_csv(Config.TRAIN_META)
    meta_val = pd.read_csv(Config.VAL_META)
    meta_test = pd.read_csv(Config.TEST_META)

    # 3. Process Splits
    logger.info("Processing Training Split...")
    df_train = _process_split(meta_train, raw_train_data, is_test=False)

    logger.info("Processing Validation Split...")
    df_val = _process_split(meta_val, raw_train_data, is_test=False)

    logger.info("Processing Test Split...")
    df_test = _process_split(meta_test, raw_test_data, is_test=True)

    # 4. Save to Cache
    logger.info(f"Saving processed datasets to {cache_dir}...")
    try:
        df_train.to_parquet(train_cache, index=False)
        df_val.to_parquet(val_cache, index=False)
        df_test.to_parquet(test_cache, index=False)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

    # 5. Debug Slicing (Post-processing)
    if Config.DEBUG:
        logger.info(
            f"Debug mode enabled. Slicing datasets to {Config.DEBUG_SIZE} samples."
        )
        df_train = df_train.head(Config.DEBUG_SIZE)
        df_val = df_val.head(Config.DEBUG_SIZE)
        df_test = df_test.head(Config.DEBUG_SIZE)

    logger.info(
        f"Data Loaded. Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
    )
    return df_train, df_val, df_test


def _process_split(
    metadata_df: pd.DataFrame, raw_data_list: list, is_test: bool
) -> pd.DataFrame:
    """
    Internal helper to merge metadata with raw JSON data and extract features.

    Args:
        metadata_df (pd.DataFrame): Metadata containing 'sample_index' and 'request_id'.
        raw_data_list (list): List of dictionaries from the raw JSON file.
        is_test (bool): Whether this is the test set (excludes target variable).

    Returns:
        pd.DataFrame: Processed DataFrame with features.
    """
    records = []

    # Iterate through metadata to retrieve corresponding raw data
    for _, row in metadata_df.iterrows():
        idx = int(row["sample_index"])
        raw_entry = raw_data_list[idx]

        # Verify alignment (sanity check)
        if raw_entry["request_id"] != row["request_id"]:
            raise ValueError(
                f"ID Mismatch at index {idx}: Meta {row['request_id']} vs Raw {raw_entry['request_id']}"
            )

        # --- Feature Extraction ---

        # 1. Text View: Concatenate Title and Text
        # Handle potential missing values with empty strings
        title = raw_entry.get("request_title", "")
        if title is None:
            title = ""

        text_body = raw_entry.get("request_text_edit_aware", "")
        if text_body is None:
            text_body = ""

        text_combined = f"{title} {text_body}".strip()

        # 2. User Persona View: Join Subreddits
        # LSA requires a "document" of terms. We treat subreddits as terms.
        subreddits = raw_entry.get(Config.SUBREDDIT_COL, [])
        if not isinstance(subreddits, list):
            subreddits = []
        subreddit_string = " ".join([str(s) for s in subreddits])

        # 3. Robust Metadata View: Numerical Columns
        # Extract columns defined in Config
        features = {
            "request_id": row["request_id"],
            "text_combined": text_combined,
            "subreddit_string": subreddit_string,
        }

        for col in Config.NUMERICAL_COLS:
            val = raw_entry.get(col, 0.0)
            # Ensure basic numeric type
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = 0.0
            features[col] = val

        # 4. Target Variable
        if not is_test:
            # Target should be in metadata, but can also verify against raw if needed
            # We strictly use metadata label as ground truth
            features["requester_received_pizza"] = int(row["requester_received_pizza"])

        records.append(features)

    return pd.DataFrame(records)
