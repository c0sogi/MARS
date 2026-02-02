import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from library.config import Config


def add_meta_features(df):
    """
    Computes meta-features for the text data.
    Specifically, calculates the natural log of the character length (plus 1).

    Args:
        df (pd.DataFrame): DataFrame containing a 'text' column.

    Returns:
        pd.DataFrame: DataFrame with the new 'log_char_len' column.
    """
    # Ensure text is string and handle potential NaNs
    texts = df["text"].fillna("").astype(str)

    # Calculate character length
    char_lens = texts.apply(len)

    # Log transform: log(x + 1) to handle potential zero lengths and reduce skew
    df["log_char_len"] = np.log1p(char_lens)

    return df


def load_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Loads the training, validation, and test datasets.
    Handles caching to parquet files to speed up subsequent runs.
    Applies target encoding and meta-feature extraction.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug (bool): If True, returns a small subset of the data for debugging.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        # Load from cache
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        # Load raw data from metadata directory
        train_df = pd.read_csv(Config.TRAIN_FILE)
        val_df = pd.read_csv(Config.VAL_FILE)
        test_df = pd.read_csv(Config.TEST_FILE)

        # Target Mapping
        # EAP: 0, HPL: 1, MWS: 2
        target_mapping = {"EAP": 0, "HPL": 1, "MWS": 2}

        if "author" in train_df.columns:
            train_df["target"] = train_df["author"].map(target_mapping)
        if "author" in val_df.columns:
            val_df["target"] = val_df["author"].map(target_mapping)

        # Add Meta Features
        train_df = add_meta_features(train_df)
        val_df = add_meta_features(val_df)
        test_df = add_meta_features(test_df)

        # Save to cache
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    # Handle Debug Mode
    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    return train_df, val_df, test_df


def get_cv_folds(n_splits=Config.N_FOLDS, random_state=Config.SEED):
    """
    Returns a StratifiedKFold iterator for cross-validation.

    Args:
        n_splits (int): Number of folds.
        random_state (int): Random seed.

    Returns:
        StratifiedKFold object.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
