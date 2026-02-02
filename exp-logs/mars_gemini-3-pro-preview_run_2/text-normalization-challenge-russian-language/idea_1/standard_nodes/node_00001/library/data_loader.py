import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    COL_SENTENCE_ID,
    COL_TOKEN_ID,
    COL_BEFORE,
    COL_AFTER,
    COL_ID,
    TOKEN_BOS,
    TOKEN_EOS,
    WORKING_DIR,
)
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("data_loader")


def load_and_process_data(split="train", load_cached_data=True, limit=None):
    """
    Loads and processes the dataset for the given split.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        limit (int, optional): Limit the number of rows for debugging.

    Returns:
        pd.DataFrame: Processed dataframe with 'prev', 'curr', 'next', and 'after' (if available).
    """
    # Determine file paths based on split
    if split == "train":
        raw_path = TRAIN_DATA_PATH
        cache_path = TRAIN_CACHE_PATH
    elif split == "val":
        raw_path = VAL_DATA_PATH
        cache_path = VAL_CACHE_PATH
    elif split == "test":
        raw_path = TEST_DATA_PATH
        cache_path = TEST_CACHE_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached data from {cache_path}...")
        df = pd.read_parquet(cache_path)
        if limit is not None:
            df = df.head(limit)
        return df

    # 2. Process from scratch
    logger.info(f"Cache not found or ignored. Processing raw data from {raw_path}...")

    # Ensure working directory exists for cache
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Load raw CSV
    # Specifying dtypes helps memory usage, though pandas usually infers well.
    # We ensure IDs are treated correctly.
    df = pd.read_csv(raw_path)

    if limit is not None:
        logger.info(f"Limiting raw data to {limit} rows.")
        df = df.head(limit)

    # Ensure string columns are strings (handle NaNs which might appear as float nan)
    df[COL_BEFORE] = df[COL_BEFORE].fillna("").astype(str)
    if COL_AFTER in df.columns:
        df[COL_AFTER] = df[COL_AFTER].fillna("").astype(str)

    # Sort to ensure sequence order
    # GroupShuffleSplit preserves order usually, but we enforce it to be safe for shifting
    df.sort_values(by=[COL_SENTENCE_ID, COL_TOKEN_ID], inplace=True)

    # Reset index to ensure alignment after sort
    df.reset_index(drop=True, inplace=True)

    logger.info("Generating N-gram features (Vectorized)...")

    # ==========================================
    # Vectorized Context Extraction
    # ==========================================
    # We want triplets: (prev, curr, next)
    # We can use shift(), but we must respect sentence boundaries.

    # 1. Create shifted columns for tokens
    # prev_token: shift(1) moves data down, so row i gets i-1
    # next_token: shift(-1) moves data up, so row i gets i+1
    df["prev"] = df[COL_BEFORE].shift(1)
    df["next"] = df[COL_BEFORE].shift(-1)

    # 2. Create shifted columns for sentence_ids to detect boundaries
    df["prev_sent"] = df[COL_SENTENCE_ID].shift(1)
    df["next_sent"] = df[COL_SENTENCE_ID].shift(-1)

    # 3. Apply boundaries
    # If current sentence_id != prev_sent, then 'prev' should be <BOS>
    # This handles the first token of every sentence.
    # Also handles the very first row where prev_sent is NaN.
    is_start = df[COL_SENTENCE_ID] != df["prev_sent"]
    df.loc[is_start, "prev"] = TOKEN_BOS

    # If current sentence_id != next_sent, then 'next' should be <EOS>
    # This handles the last token of every sentence.
    # Also handles the very last row where next_sent is NaN.
    is_end = df[COL_SENTENCE_ID] != df["next_sent"]
    df.loc[is_end, "next"] = TOKEN_EOS

    # Rename 'before' to 'curr' for consistency with logic
    df["curr"] = df[COL_BEFORE]

    # Select final columns
    cols_to_keep = ["prev", "curr", "next"]

    # Append target if exists
    if COL_AFTER in df.columns:
        cols_to_keep.append(COL_AFTER)

    # Append ID if exists (needed for submission)
    if COL_ID in df.columns:
        cols_to_keep.append(COL_ID)
    # If ID column doesn't exist (e.g. in train metadata), we might construct it or ignore it.
    # The train metadata usually has sentence_id and token_id.
    # The test metadata has 'id' implicitly or we construct it for submission?
    # Looking at metadata/test.csv schema in prompt: sentence_id, token_id, before.
    # The prompt says: "id column used in submission... formed by concatenating sentence_id and token_id"
    # Let's ensure 'id' exists for test set processing.
    if split == "test" and COL_ID not in df.columns:
        df[COL_ID] = (
            df[COL_SENTENCE_ID].astype(str) + "_" + df[COL_TOKEN_ID].astype(str)
        )
        cols_to_keep.append(COL_ID)

    # Filter DataFrame
    df_processed = df[cols_to_keep].copy()

    # Save to cache
    logger.info(f"Saving processed data to {cache_path}...")
    df_processed.to_parquet(cache_path, index=False)

    return df_processed
