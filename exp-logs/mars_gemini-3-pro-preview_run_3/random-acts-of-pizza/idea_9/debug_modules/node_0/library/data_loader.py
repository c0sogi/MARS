import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed


def load_dataset(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.
    Performs initial cleaning, leakage removal, and caching.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
                                 If False or cache miss, re-processes and saves to cache.

    Returns:
        tuple: (train_df, val_df, test_df)
            - train_df (pd.DataFrame): Cleaned training data with target.
            - val_df (pd.DataFrame): Cleaned validation data with target.
            - test_df (pd.DataFrame): Cleaned test data without target.
    """
    set_seed()

    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_processed.parquet")
    val_cache_path = os.path.join(cache_dir, "val_processed.parquet")
    test_cache_path = os.path.join(cache_dir, "test_processed.parquet")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            print("Loading datasets from cache...")
            try:
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing data.")
        else:
            print("Cache not found. Processing raw metadata...")
    else:
        print("Ignoring cache. Processing raw metadata...")

    # Load raw metadata
    print(f"Loading raw data from {Config.METADATA_DIR}...")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    # Helper function for cleaning
    def clean_dataframe(df, is_test=False):
        # 1. Drop Leakage Columns (ending in _at_retrieval)
        # These columns contain future information not available at prediction time
        leakage_cols = [c for c in df.columns if c.endswith("_at_retrieval")]
        if leakage_cols:
            df = df.drop(columns=leakage_cols)

        # 2. Text Cleaning
        # Ensure request_text_edit_aware is used as the primary text source
        # If it's missing (NaN), fall back to request_text, then empty string
        if "request_text_edit_aware" in df.columns:
            df["request_text_edit_aware"] = (
                df["request_text_edit_aware"]
                .fillna(df["request_text"] if "request_text" in df.columns else "")
                .fillna("")
            )

        # Ensure title is not NaN
        if "request_title" in df.columns:
            df["request_title"] = df["request_title"].fillna("")

        # 3. Handle Subreddits List
        # Ensure the column exists and handle potential NaNs (though unlikely in raw json)
        if "requester_subreddits_at_request" in df.columns:
            # If it's NaN, make it an empty list
            # Note: In parquet, list columns can be tricky if they have NaNs.
            # We fill with empty lists where necessary.
            # We assume the column is object type containing lists.
            missing_mask = df["requester_subreddits_at_request"].isnull()
            if missing_mask.any():
                df.loc[missing_mask, "requester_subreddits_at_request"] = df.loc[
                    missing_mask, "requester_subreddits_at_request"
                ].apply(lambda x: [])

        return df

    # Apply cleaning
    print("Cleaning datasets...")
    train_df = clean_dataframe(train_df, is_test=False)
    val_df = clean_dataframe(val_df, is_test=False)
    test_df = clean_dataframe(test_df, is_test=True)

    # Save to cache
    print(f"Saving processed datasets to {cache_dir}...")
    train_df.to_parquet(train_cache_path, index=False)
    val_df.to_parquet(val_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, val_df, test_df
