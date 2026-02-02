import os
import string
import pandas as pd
import numpy as np
from library.config import Config


def extract_meta_features(df):
    """
    Extracts explicit meta-features from the 'text' column of a DataFrame.

    Features:
    - char_len: Number of characters in the sentence.
    - word_count: Number of words in the sentence.
    - punct_density: Ratio of punctuation characters to total characters.

    Args:
        df (pd.DataFrame): DataFrame containing a 'text' column.

    Returns:
        pd.DataFrame: DataFrame with the extracted features.
    """
    # Ensure text column exists
    if "text" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'text' column.")

    # Fill NaNs and ensure string type
    texts = df["text"].fillna("").astype(str)

    # 1. Sentence Character Length
    char_len = texts.apply(len)

    # 2. Word Count
    # Using simple whitespace splitting
    word_count = texts.apply(lambda x: len(x.split()))

    # 3. Punctuation Density
    punct_set = set(string.punctuation)

    def count_punctuation(text):
        return sum(1 for char in text if char in punct_set)

    punct_counts = texts.apply(count_punctuation)

    # Calculate density, handling division by zero (replace 0 length with 1 to avoid error)
    # If char_len is 0, the numerator is also 0, so result is 0.
    punct_density = punct_counts / char_len.replace(0, 1)

    # Assemble features
    meta_features = pd.DataFrame(
        {"char_len": char_len, "word_count": word_count, "punct_density": punct_density}
    )

    return meta_features


def get_meta_features(
    data_path, cache_path, load_cached_data=True, debug=False, sample_size=100
):
    """
    Loads data, extracts meta-features, and handles caching.

    Args:
        data_path (str): Path to the input CSV file.
        cache_path (str): Path to the parquet file for caching.
        load_cached_data (bool): If True, attempts to load from cache first.
        debug (bool): If True, processes only a subset of data and uses a debug cache file.
        sample_size (int): Number of rows to process if debug is True.

    Returns:
        pd.DataFrame: DataFrame containing the meta-features.
    """
    # Adjust cache path for debug mode to prevent polluting the main cache
    if debug:
        base, ext = os.path.splitext(cache_path)
        cache_path = f"{base}_debug{ext}"

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            # If loading fails (e.g., corrupt file), proceed to recompute
            pass

    # 2. Load raw data and compute features
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Load data (subset if debug)
    if debug:
        df = pd.read_csv(data_path, nrows=sample_size)
    else:
        df = pd.read_csv(data_path)

    features = extract_meta_features(df)

    # 3. Save to cache
    features.to_parquet(cache_path, index=False)

    return features
