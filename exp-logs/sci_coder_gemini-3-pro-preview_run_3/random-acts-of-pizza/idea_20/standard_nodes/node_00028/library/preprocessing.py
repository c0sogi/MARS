import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    TEXT_EDIT_AWARE_COL,
    SUBREDDIT_COL,
)


def clean_text(text):
    """
    Cleans the edit-aware text by handling NaNs and whitespace.
    The input column 'request_text_edit_aware' is already stripped of edits
    by the dataset provider, so this function ensures valid string format.
    """
    if pd.isna(text):
        return ""
    return str(text).strip()


def serialize_subreddits(subreddits):
    """
    Converts a list of subreddits into a space-separated string.
    Handles lists, numpy arrays, and existing strings.
    """
    if subreddits is None or (isinstance(subreddits, float) and np.isnan(subreddits)):
        return ""

    # If it's a list or numpy array
    if isinstance(subreddits, (list, np.ndarray)):
        # Filter out None or empty strings
        valid_subs = [str(s) for s in subreddits if s]
        return " ".join(valid_subs)

    # If it's already a string
    if isinstance(subreddits, str):
        return subreddits.strip()

    return str(subreddits)


def load_dataset(load_cached_data=True):
    """
    Loads the dataset (Train, Val, Test).
    Implements caching to avoid re-processing text and lists.

    Args:
        load_cached_data (bool): If True, tries to load from ./working/idea_20/
                                 before processing from scratch.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    train_cache = os.path.join(WORKING_DIR, "train_cleaned.parquet")
    val_cache = os.path.join(WORKING_DIR, "val_cleaned.parquet")
    test_cache = os.path.join(WORKING_DIR, "test_cleaned.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading preprocessed data from cache...")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Error loading cache: {e}. Re-processing...")
        else:
            print("Cache files not found. Processing from scratch...")
    else:
        print("Skipping cache. Processing from scratch...")

    # 2. Process from scratch
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    print("Loading metadata...")
    # Load raw metadata
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Metadata file not found: {TRAIN_PATH}")

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print("Preprocessing text and subreddits...")

    # Apply cleaning functions
    for df in [train_df, val_df, test_df]:
        # Process Text
        if TEXT_EDIT_AWARE_COL in df.columns:
            df[TEXT_EDIT_AWARE_COL] = df[TEXT_EDIT_AWARE_COL].apply(clean_text)

        # Process Subreddits
        if SUBREDDIT_COL in df.columns:
            df[SUBREDDIT_COL] = df[SUBREDDIT_COL].apply(serialize_subreddits)

    # 3. Save to cache
    print("Saving preprocessed data to cache...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df
