import os
import pandas as pd
import numpy as np
from library.config import Config


def preprocess_subreddits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Serializes the subreddit list column into a space-separated string column.
    This transformation is necessary for both the 'Community Bagger' (TF-IDF)
    and 'Persona Booster' (MPNet Embeddings) to treat user history as text.

    Args:
        df: Input DataFrame containing the subreddit column defined in Config.

    Returns:
        DataFrame with the subreddit column converted to space-separated strings.
    """
    col = Config.SUBREDDIT_COL

    if col not in df.columns:
        return df

    def serialize_list(item):
        if item is None:
            return ""
        if isinstance(item, (list, np.ndarray)):
            # Filter None values and join strings
            return " ".join([str(s) for s in item if s is not None])
        if isinstance(item, str):
            # Already a string, return as is
            return item
        return str(item)

    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    df_processed[col] = df_processed[col].apply(serialize_list)

    return df_processed


def load_datasets(load_cached_data: bool = True, sample_size: int = None):
    """
    Loads train, validation, and test datasets.
    Implements a caching mechanism for the preprocessed (subreddit-serialized) data.

    Args:
        load_cached_data: If True, attempts to load preprocessed data from the cache directory.
                          If False or if cache files are missing, loads from raw metadata,
                          processes the data, and saves to cache.
        sample_size:      If provided, restricts the returned DataFrames to the first N rows.
                          Useful for debugging and rapid iteration.

    Returns:
        Tuple containing (train_df, val_df, test_df).
    """
    # Define cache file paths
    train_cache_path = os.path.join(Config.CACHE_DIR, "train_base.parquet")
    val_cache_path = os.path.join(Config.CACHE_DIR, "val_base.parquet")
    test_cache_path = os.path.join(Config.CACHE_DIR, "test_base.parquet")

    # Ensure working/cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    )

    if load_cached_data and cache_exists:
        print("Loading datasets from cache...")
        train_df = pd.read_parquet(train_cache_path)
        val_df = pd.read_parquet(val_cache_path)
        test_df = pd.read_parquet(test_cache_path)
    else:
        print("Loading datasets from metadata and processing...")

        # Verify metadata availability
        if not os.path.exists(Config.TRAIN_PATH):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_PATH}"
            )
        if not os.path.exists(Config.VAL_PATH):
            raise FileNotFoundError(
                f"Validation metadata not found at {Config.VAL_PATH}"
            )
        if not os.path.exists(Config.TEST_PATH):
            raise FileNotFoundError(f"Test metadata not found at {Config.TEST_PATH}")

        # Load raw metadata
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        val_df = pd.read_parquet(Config.VAL_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)

        # Apply preprocessing (Subreddit Serialization)
        train_df = preprocess_subreddits(train_df)
        val_df = preprocess_subreddits(val_df)
        test_df = preprocess_subreddits(test_df)

        # Save to cache for future runs
        print(f"Saving processed datasets to {Config.CACHE_DIR}...")
        train_df.to_parquet(train_cache_path, index=False)
        val_df.to_parquet(val_cache_path, index=False)
        test_df.to_parquet(test_cache_path, index=False)

    # Apply sampling if requested (for debugging)
    if sample_size is not None:
        print(f"Sampling datasets to {sample_size} rows for debugging...")
        train_df = train_df.head(sample_size)
        val_df = val_df.head(sample_size)
        test_df = test_df.head(sample_size)

    return train_df, val_df, test_df
