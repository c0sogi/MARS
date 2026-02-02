import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    TARGET_COL,
    ID_COL,
    DEBUG_SAMPLE_SIZE,
    SEED,
)
from library.utils import set_seed


def load_dataset(load_cached_data: bool = True):
    """
    Loads, cleans, and preprocesses the dataset.

    Args:
        load_cached_data (bool): If True, attempts to load processed files from the working directory.
                                 If False or files don't exist, processes from scratch and saves cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    set_seed(SEED)

    # Define cache file paths
    train_cache_path = os.path.join(WORKING_DIR, "train_cleaned.parquet")
    val_cache_path = os.path.join(WORKING_DIR, "val_cleaned.parquet")
    test_cache_path = os.path.join(WORKING_DIR, "test_cleaned.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            print("Loading cleaned data from cache...")
            train_df = pd.read_parquet(train_cache_path)
            val_df = pd.read_parquet(val_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            return train_df, val_df, test_df
        else:
            print("Cache not found. Processing from raw metadata...")
    else:
        print("Ignoring cache. Processing from raw metadata...")

    # 2. Load raw data
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Train metadata not found at {TRAIN_PATH}")
    if not os.path.exists(VAL_PATH):
        raise FileNotFoundError(f"Val metadata not found at {VAL_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_PATH}")

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # 3. Preprocessing and Cleaning
    def clean_dataframe(df, is_test=False):
        # A. Drop Leakage Columns (_at_retrieval)
        # These columns contain data collected long after the request was made
        leakage_cols = [c for c in df.columns if c.endswith("_at_retrieval")]
        if leakage_cols:
            df = df.drop(columns=leakage_cols)

        # B. Standardize Text Columns
        # Use 'request_text_edit_aware' if available to avoid edit-based leakage
        # Fallback to 'request_text' if necessary (though metadata ensures edit_aware exists)
        if "request_text_edit_aware" in df.columns:
            df["text"] = df["request_text_edit_aware"].fillna("").astype(str)
        elif "request_text" in df.columns:
            df["text"] = df["request_text"].fillna("").astype(str)
        else:
            df["text"] = ""

        # Standardize Title
        if "request_title" in df.columns:
            df["title"] = df["request_title"].fillna("").astype(str)
        else:
            df["title"] = ""

        # C. Ensure ID column is string
        if ID_COL in df.columns:
            df[ID_COL] = df[ID_COL].astype(str)

        # D. Handle Target (only for train/val)
        if not is_test and TARGET_COL in df.columns:
            df[TARGET_COL] = df[TARGET_COL].astype(int)

        return df

    print("Cleaning training data...")
    train_df = clean_dataframe(train_df, is_test=False)

    print("Cleaning validation data...")
    val_df = clean_dataframe(val_df, is_test=False)

    print("Cleaning test data...")
    test_df = clean_dataframe(test_df, is_test=True)

    # 4. Debug Sampling
    if DEBUG_SAMPLE_SIZE is not None:
        print(f"Debug mode enabled. Subsampling to {DEBUG_SAMPLE_SIZE} rows.")
        if len(train_df) > DEBUG_SAMPLE_SIZE:
            train_df = train_df.sample(
                n=DEBUG_SAMPLE_SIZE, random_state=SEED
            ).reset_index(drop=True)
        if len(val_df) > DEBUG_SAMPLE_SIZE:
            val_df = val_df.sample(n=DEBUG_SAMPLE_SIZE, random_state=SEED).reset_index(
                drop=True
            )
        # We also subsample test to ensure the pipeline runs quickly end-to-end during debug
        if len(test_df) > DEBUG_SAMPLE_SIZE:
            test_df = test_df.sample(
                n=DEBUG_SAMPLE_SIZE, random_state=SEED
            ).reset_index(drop=True)

    # 5. Save to Cache
    print(f"Saving cleaned data to {WORKING_DIR}...")
    os.makedirs(WORKING_DIR, exist_ok=True)
    train_df.to_parquet(train_cache_path, index=False)
    val_df.to_parquet(val_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, val_df, test_df
