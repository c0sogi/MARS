import os
import random
import numpy as np
import pandas as pd
from library.config import TRAIN_PATH, VAL_PATH, TEST_PATH, CACHE_DIR, SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and pandas.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def preprocess(text):
    """
    Preprocesses the text by converting it to a string and lowercasing it.
    """
    return str(text).lower()


def tokenize(text):
    """
    Tokenizes the text by splitting on whitespace.
    This preserves punctuation attached to words, as required for the task.
    """
    return str(text).split()


def jaccard(str1, str2):
    """
    Calculates the word-level Jaccard similarity score between two strings.
    """
    a = set(tokenize(preprocess(str1)))
    b = set(tokenize(preprocess(str2)))

    # If both sets are empty, they are identical (perfect match)
    if len(a) == 0 and len(b) == 0:
        return 1.0

    c = a.intersection(b)
    union_len = len(a) + len(b) - len(c)

    if union_len == 0:
        return 0.0

    return float(len(c)) / float(union_len)


def load_processed_data(load_cached_data=True, debug=False, debug_size=500):
    """
    Loads the train, validation, and test datasets.
    Applies preprocessing and caching mechanisms using Parquet files.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, returns a subset of the data.
        debug_size (int): Number of rows to return if debug is True.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache file paths
    train_cache_path = os.path.join(CACHE_DIR, "train_processed.parquet")
    val_cache_path = os.path.join(CACHE_DIR, "val_processed.parquet")
    test_cache_path = os.path.join(CACHE_DIR, "test_processed.parquet")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_df = None
    val_df = None
    test_df = None
    loaded_from_cache = False

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            try:
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                loaded_from_cache = True
            except Exception:
                # If loading fails, proceed to recompute
                loaded_from_cache = False

    # 2. If not loaded, process from scratch
    if not loaded_from_cache:
        # Load raw metadata
        train_df = pd.read_csv(TRAIN_PATH)
        val_df = pd.read_csv(VAL_PATH)
        test_df = pd.read_csv(TEST_PATH)

        # Ensure string types
        train_df["text"] = train_df["text"].astype(str)
        train_df["selected_text"] = train_df["selected_text"].astype(str)
        val_df["text"] = val_df["text"].astype(str)
        val_df["selected_text"] = val_df["selected_text"].astype(str)
        test_df["text"] = test_df["text"].astype(str)

        # Apply preprocessing to create clean columns
        # This allows the model to work with normalized text while keeping original for reference
        train_df["text_clean"] = train_df["text"].apply(preprocess)
        train_df["selected_text_clean"] = train_df["selected_text"].apply(preprocess)

        val_df["text_clean"] = val_df["text"].apply(preprocess)
        val_df["selected_text_clean"] = val_df["selected_text"].apply(preprocess)

        test_df["text_clean"] = test_df["text"].apply(preprocess)

        # Save to cache
        train_df.to_parquet(train_cache_path)
        val_df.to_parquet(val_cache_path)
        test_df.to_parquet(test_cache_path)

    # 3. Handle Debug mode (Subsampling)
    if debug:
        train_df = train_df.iloc[:debug_size].reset_index(drop=True)
        val_df = val_df.iloc[:debug_size].reset_index(drop=True)
        test_df = test_df.iloc[:debug_size].reset_index(drop=True)

    return train_df, val_df, test_df
