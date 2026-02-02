import os
import pandas as pd
import numpy as np
from library.config import (
    METADATA_DIR,
    WORKING_DIR,
    METADATA_FEATURES,
    TEXT_COLS,
    TARGET_COL,
    ID_COL,
    SUBREDDIT_COL,
)


def clean_and_select_features(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Performs hygienic feature selection, text concatenation, and leakage removal.

    Args:
        df: Raw DataFrame loaded from metadata.
        is_test: Boolean indicating if this is the test set (target column might be missing).

    Returns:
        Cleaned DataFrame with specific columns.
    """
    # 1. Text Concatenation (Title + Edit-Aware Body)
    # Ensure text columns are strings and handle NaNs
    title_col = TEXT_COLS[0]
    body_col = TEXT_COLS[1]

    df[title_col] = df[title_col].fillna("").astype(str)
    df[body_col] = df[body_col].fillna("").astype(str)

    # Create combined text column for downstream vectorization
    df["text_combined"] = df[title_col] + " " + df[body_col]

    # 2. Subreddit History Processing
    # Convert list of subreddits to space-separated string for Bag-of-Concepts Vectorization
    if SUBREDDIT_COL in df.columns:

        def process_subreddits(val):
            if isinstance(val, (list, np.ndarray)):
                # Join list elements with spaces
                return " ".join([str(s) for s in val])
            elif isinstance(val, str):
                return val
            return ""

        df["subreddit_string"] = df[SUBREDDIT_COL].apply(process_subreddits)
    else:
        # Fallback if column is missing (though it should be in metadata)
        df["subreddit_string"] = ""

    # 3. Feature Selection (Allow-List)
    # We strictly select only the columns defined in the configuration to prevent
    # any retrieval-time leakage (e.g., upvotes_at_retrieval).
    keep_cols = [ID_COL] + METADATA_FEATURES + ["text_combined", "subreddit_string"]

    if not is_test:
        if TARGET_COL in df.columns:
            keep_cols.append(TARGET_COL)
        else:
            raise ValueError(
                f"Target column '{TARGET_COL}' missing in training/validation data."
            )

    # Filter the DataFrame to keep only the allow-listed columns
    final_df = df[keep_cols].copy()

    return final_df


def load_dataset(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads the dataset for a specific split (train, val, test), performs preprocessing,
    and handles caching.

    Args:
        split: One of 'train', 'val', 'test'.
        load_cached_data: If True, attempts to load from cache first.

    Returns:
        Processed DataFrame.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}.")

    # Define cache path
    cache_filename = f"processed_{split}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load raw metadata
    metadata_filename = f"{split}.parquet"
    metadata_path = os.path.join(METADATA_DIR, metadata_filename)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    print(f"Loading raw {split} data from {metadata_path}...")
    raw_df = pd.read_parquet(metadata_path)

    # 3. Process data
    is_test = split == "test"
    processed_df = clean_and_select_features(raw_df, is_test=is_test)

    # 4. Save to cache
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving processed {split} data to {cache_path}...")
    processed_df.to_parquet(cache_path, index=False)

    return processed_df
