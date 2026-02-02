import os
import random
import codecs
import numpy as np
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and hash functions.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def decode_text(text):
    """
    Decodes unicode-escaped text (e.g., converting '\\u0027' to "'").
    Handles NaNs and exceptions gracefully.

    Args:
        text (str or object): The text to decode.

    Returns:
        str: The decoded text.
    """
    if pd.isna(text):
        return ""
    try:
        # Decode python byte literal representation if present
        return codecs.decode(str(text), "unicode_escape")
    except Exception:
        return str(text)


def load_data(data_type, load_cached_data=True, max_samples=None):
    """
    Loads the dataset specified by data_type, applies text decoding, and handles caching.

    Logic:
    1. Checks if a cached parquet file exists in the working directory.
    2. If load_cached_data is True and file exists, loads and returns it.
    3. Otherwise, loads raw CSV, applies decode_text, saves to parquet, and returns it.

    Args:
        data_type (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        max_samples (int, optional): If provided, limits the number of rows returned.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Determine input paths and cache filenames based on data_type
    if data_type == "train":
        input_path = Config.TRAIN_DATA_PATH
        cache_filename = "train_decoded.parquet"
    elif data_type == "val":
        input_path = Config.VAL_DATA_PATH
        cache_filename = "val_decoded.parquet"
    elif data_type == "test":
        input_path = Config.TEST_DATA_PATH
        cache_filename = "test_decoded.parquet"
    else:
        raise ValueError("data_type must be one of 'train', 'val', 'test'")

    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    df = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
        except Exception:
            # If loading fails (e.g., corrupt file), proceed to process from scratch
            df = None

    # 2. Process from scratch if not loaded
    if df is None:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path)

        # Apply deterministic text processing
        if Config.TEXT_COL in df.columns:
            df[Config.TEXT_COL] = df[Config.TEXT_COL].apply(decode_text)

        # Save to cache (Parquet format)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    # 3. Apply dataset size limit if requested (e.g., for debugging)
    if max_samples is not None:
        df = df.head(max_samples)

    return df
