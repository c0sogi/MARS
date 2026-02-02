import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything


def preprocess_text(text):
    """
    Applies basic preprocessing to the text.

    Args:
        text (str): The input text.

    Returns:
        str: The preprocessed (lowercased) text.
    """
    if pd.isna(text):
        return ""
    return str(text).lower()


def load_data(split="train", load_cached_data=True, nrows=None):
    """
    Loads and preprocesses data for a specific split. Implements caching to Parquet.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.
        nrows (int, optional): Number of rows to return (useful for debugging).

    Returns:
        pd.DataFrame: The loaded and processed dataframe.
    """
    seed_everything()

    # Determine raw file path based on split
    if split == "train":
        raw_path = Config.TRAIN_DATA_PATH
    elif split == "val":
        raw_path = Config.VAL_DATA_PATH
    elif split == "test":
        raw_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache path
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_processed.parquet")

    df = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load full cached dataframe
            df = pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Error loading cache for {split}: {e}. Proceeding to re-process.")
            df = None

    # 2. If not loaded (or cache disabled), process from scratch
    if df is None:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found at {raw_path}")

        # Load raw data
        df = pd.read_csv(raw_path)

        # Apply preprocessing
        if Config.TEXT_COL in df.columns:
            df[Config.TEXT_COL] = df[Config.TEXT_COL].apply(preprocess_text)
        else:
            raise KeyError(f"Column '{Config.TEXT_COL}' not found in {split} data.")

        # Save to cache (save the full processed dataset)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")

    # 3. Apply dataset size limit if requested
    if nrows is not None:
        df = df.head(nrows)

    return df
