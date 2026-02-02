import os
import re
import pandas as pd
import numpy as np
from library.config import Config


def clean_text(text):
    """
    Strips edit-aware content using regex.
    Removes common post-hoc edit patterns such as 'EDIT:', 'Update:', etc.,
    which might leak the outcome or introduce noise.
    """
    if not isinstance(text, str):
        return ""

    # Regex to identify and remove text starting with "Edit:" or "Update:"
    # (case insensitive) until the end of the string, often found at the end of posts.
    # We use DOTALL so . matches newlines if the edit block is multi-line.
    pattern = r"(?i)\n\s*(?:edit|update)\s*:?.*$"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)

    return cleaned_text.strip()


def serialize_subreddits(subreddits):
    """
    Converts a list of subreddits into a space-separated string for TF-IDF vectorization.
    Handles lists, numpy arrays, and potential missing values.
    """
    if isinstance(subreddits, (list, np.ndarray)):
        # Filter out None or empty strings and join with space
        valid_subs = [str(s) for s in subreddits if s]
        return " ".join(valid_subs)
    elif isinstance(subreddits, str):
        # If already a string (e.g. from a CSV reload), return as is
        return subreddits

    return ""


def load_data(split, debug_sample_size=None):
    """
    Loads the raw metadata Parquet file for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug_sample_size (int, optional): Override Config.DEBUG_SAMPLE_SIZE.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split provided: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_parquet(path)

    # Handle debug sampling
    sample_size = (
        debug_sample_size if debug_sample_size is not None else Config.DEBUG_SAMPLE_SIZE
    )
    if sample_size is not None and sample_size < len(df):
        df = df.iloc[:sample_size].copy()

    return df


def get_processed_data(split, load_cached_data=True, debug_sample_size=None):
    """
    Loads data, applies cleaning and serialization, and utilizes caching.

    This function fulfills the requirement for deterministic data processing with caching.
    It checks for a cached Parquet file in the Config.CACHE_DIR. If found and load_cached_data is True,
    it loads it. Otherwise, it loads raw data, processes it, saves to cache, and returns it.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_sample_size (int, optional): Override Config.DEBUG_SAMPLE_SIZE.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Determine effective sample size for cache naming to avoid collisions
    current_sample_size = (
        debug_sample_size if debug_sample_size is not None else Config.DEBUG_SAMPLE_SIZE
    )
    size_suffix = (
        f"_sample{current_sample_size}" if current_sample_size is not None else ""
    )

    cache_filename = f"{split}_processed{size_suffix}.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading processed {split} data from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Error loading cache ({e}). Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {split} data from scratch...")
    df = load_data(split, debug_sample_size)

    # Apply Text Cleaning
    text_col = Config.TEXT_COL
    if text_col in df.columns:
        # Fill NaNs before processing
        df[text_col] = df[text_col].fillna("").apply(clean_text)

    # Apply Subreddit Serialization
    sub_col = Config.SUBREDDIT_COL
    if sub_col in df.columns:
        df[sub_col] = df[sub_col].apply(serialize_subreddits)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved processed {split} data to cache: {cache_path}")

    return df
