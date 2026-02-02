import os
import re
import pandas as pd
import numpy as np
from library.config import Config


def clean_text(text):
    """
    Cleans text by normalizing whitespace and handling missing values.

    Args:
        text (str or pd.NA): Input text.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Replace newlines, tabs, and multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def flatten_subreddits(subreddits):
    """
    Converts a list of subreddits into a space-separated string.

    Args:
        subreddits (list, np.ndarray, or str): List of subreddit names.

    Returns:
        str: Space-separated string of subreddits.
    """
    if isinstance(subreddits, (list, np.ndarray)):
        if len(subreddits) == 0:
            return ""
        # Join elements with space, ensuring they are strings
        return " ".join(str(s) for s in subreddits)
    elif pd.isna(subreddits):
        return ""
    else:
        # If it's already a string or other type, convert to string
        return str(subreddits)


def load_data(load_cached_data=True, debug=False):
    """
    Loads, processes, and caches the dataset.

    This function handles:
    1. Loading from cache if available and requested.
    2. Loading raw metadata if cache is not used.
    3. Preprocessing text and behavioral features.
    4. Removing leakage columns (those available only at retrieval time).
    5. Caching the processed data for future runs.
    6. Sampling data if debug mode is enabled.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): Whether to sample the data for debugging.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache file paths
    cache_train_path = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    cache_val_path = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    cache_test_path = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    data_loaded = False
    train_df, val_df, test_df = None, None, None

    # Attempt to load from cache
    if load_cached_data:
        try:
            if (
                os.path.exists(cache_train_path)
                and os.path.exists(cache_val_path)
                and os.path.exists(cache_test_path)
            ):

                print("Loading data from cache...")
                train_df = pd.read_parquet(cache_train_path)
                val_df = pd.read_parquet(cache_val_path)
                test_df = pd.read_parquet(cache_test_path)
                data_loaded = True
            else:
                print("Cache files not found. Processing from scratch...")
        except Exception as e:
            print(f"Error loading cache: {e}. Processing from scratch...")
            data_loaded = False

    # Process from scratch if cache load failed or was skipped
    if not data_loaded:
        print("Loading raw metadata...")
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        val_df = pd.read_parquet(Config.VAL_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)

        print("Preprocessing text and subreddits...")
        for df in [train_df, val_df, test_df]:
            # Clean Request Text
            if Config.TEXT_COL in df.columns:
                df[Config.TEXT_COL] = df[Config.TEXT_COL].apply(clean_text)

            # Clean Request Title
            if Config.TITLE_COL in df.columns:
                df[Config.TITLE_COL] = df[Config.TITLE_COL].apply(clean_text)

            # Flatten Subreddit List
            if Config.SUBREDDIT_COL in df.columns:
                df[Config.SUBREDDIT_COL] = df[Config.SUBREDDIT_COL].apply(
                    flatten_subreddits
                )

        print("Removing leakage features...")
        # Identify and drop columns that contain future information (retrieval time stats)
        # These are present in Train/Val but must not be used for prediction.
        leakage_cols_train = [
            c for c in train_df.columns if c.endswith(Config.RETRIEVAL_SUFFIX)
        ]
        if leakage_cols_train:
            print(f"Dropping {len(leakage_cols_train)} leakage columns from Train.")
            train_df.drop(columns=leakage_cols_train, inplace=True)

        leakage_cols_val = [
            c for c in val_df.columns if c.endswith(Config.RETRIEVAL_SUFFIX)
        ]
        if leakage_cols_val:
            val_df.drop(columns=leakage_cols_val, inplace=True)

        # Ensure Test set is also clean (though usually it doesn't have these columns)
        leakage_cols_test = [
            c for c in test_df.columns if c.endswith(Config.RETRIEVAL_SUFFIX)
        ]
        if leakage_cols_test:
            test_df.drop(columns=leakage_cols_test, inplace=True)

        print("Caching processed data...")
        try:
            # Ensure working directory exists
            os.makedirs(Config.WORKING_DIR, exist_ok=True)

            train_df.to_parquet(cache_train_path, index=False)
            val_df.to_parquet(cache_val_path, index=False)
            test_df.to_parquet(cache_test_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    # Handle Debug Mode
    if debug:
        print(f"Debug mode enabled: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    return train_df, val_df, test_df
