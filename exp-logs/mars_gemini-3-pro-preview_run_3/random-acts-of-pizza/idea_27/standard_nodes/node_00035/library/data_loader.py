import pandas as pd
import numpy as np
from library.config import Config


def clean_request_text(text):
    """
    Cleans the request text by handling missing values and ensuring string format.

    Args:
        text: The input text (str or NaN).

    Returns:
        str: Cleaned text or empty string if input is NaN.
    """
    if pd.isna(text):
        return ""
    return str(text).strip()


def serialize_subreddits(subreddits):
    """
    Converts a list of subreddits into a space-separated string for TF-IDF vectorization.

    Args:
        subreddits: List of subreddit strings or NaN.

    Returns:
        str: Space-separated string of subreddits.
    """
    if isinstance(subreddits, (list, np.ndarray)):
        # Join non-empty strings
        return " ".join([str(s) for s in subreddits if s])
    return ""


def load_data(sample_size=None):
    """
    Loads the train, validation, and test datasets from Parquet files.
    Applies text cleaning and subreddit serialization.
    Ensures the target variable is strictly integer-typed.

    Args:
        sample_size (int, optional): Number of rows to load for debugging.
                                     If None, loads full datasets.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Load datasets from Parquet
    # We load the full files first because parquet reading is efficient
    train_df = pd.read_parquet(Config.TRAIN_DATA_PATH)
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)
    test_df = pd.read_parquet(Config.TEST_DATA_PATH)

    # Apply sampling for debugging if requested
    if sample_size is not None:
        train_df = train_df.head(sample_size)
        val_df = val_df.head(sample_size)
        test_df = test_df.head(sample_size)

    # Process Training Data
    if Config.TEXT_COL in train_df.columns:
        train_df[Config.TEXT_COL] = train_df[Config.TEXT_COL].apply(clean_request_text)

    if Config.HISTORY_COL in train_df.columns:
        train_df[Config.HISTORY_COL] = train_df[Config.HISTORY_COL].apply(
            serialize_subreddits
        )

    if Config.TARGET_COL in train_df.columns:
        train_df[Config.TARGET_COL] = train_df[Config.TARGET_COL].astype(int)

    # Process Validation Data
    if Config.TEXT_COL in val_df.columns:
        val_df[Config.TEXT_COL] = val_df[Config.TEXT_COL].apply(clean_request_text)

    if Config.HISTORY_COL in val_df.columns:
        val_df[Config.HISTORY_COL] = val_df[Config.HISTORY_COL].apply(
            serialize_subreddits
        )

    if Config.TARGET_COL in val_df.columns:
        val_df[Config.TARGET_COL] = val_df[Config.TARGET_COL].astype(int)

    # Process Test Data
    if Config.TEXT_COL in test_df.columns:
        test_df[Config.TEXT_COL] = test_df[Config.TEXT_COL].apply(clean_request_text)

    if Config.HISTORY_COL in test_df.columns:
        test_df[Config.HISTORY_COL] = test_df[Config.HISTORY_COL].apply(
            serialize_subreddits
        )

    # Handle target in test if it exists
    if Config.TARGET_COL in test_df.columns:
        test_df[Config.TARGET_COL] = test_df[Config.TARGET_COL].astype(int)

    return train_df, val_df, test_df
