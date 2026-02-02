import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    RANDOM_SEED,
    TARGET_COL,
)


def clean_data(df):
    """
    Cleans the dataframe by removing leakage columns and ensuring correct text column usage.

    Args:
        df (pd.DataFrame): The dataframe to clean.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    # Identify columns to drop (leakage)
    # We drop any column ending in '_at_retrieval' as per instructions to prevent data leakage
    leakage_cols = [col for col in df.columns if col.endswith("_at_retrieval")]

    # Also drop the raw 'request_text' if 'request_text_edit_aware' exists
    # to enforce usage of the edit-aware version which removes "EDIT: Thanks" messages
    if "request_text" in df.columns and "request_text_edit_aware" in df.columns:
        leakage_cols.append("request_text")

    df_cleaned = df.drop(columns=leakage_cols, errors="ignore")

    return df_cleaned


def load_datasets(load_cached_data=True, sample_size=None):
    """
    Loads, cleans, and returns the train, validation, and test datasets.
    Implements caching to parquet files in the working directory.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        sample_size (int, optional): If provided, returns a subset of the data for debugging.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test)
    """
    # Set global seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(WORKING_DIR, "train_cleaned.parquet")
    val_cache = os.path.join(WORKING_DIR, "val_cleaned.parquet")
    test_cache = os.path.join(WORKING_DIR, "test_cleaned.parquet")

    # Check if cache exists and is requested
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cleaned datasets from cache at {WORKING_DIR}...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        print("Loading raw metadata and performing cleaning...")
        # Load from metadata
        if not os.path.exists(TRAIN_PATH):
            raise FileNotFoundError(f"Metadata file not found: {TRAIN_PATH}")

        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)
        test_df = pd.read_parquet(TEST_PATH)

        # Clean data
        train_df = clean_data(train_df)
        val_df = clean_data(val_df)
        test_df = clean_data(test_df)

        # Save to cache
        print(f"Saving cleaned datasets to cache at {WORKING_DIR}...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    # Apply sampling if requested (for debugging)
    if sample_size is not None:
        print(f"Subsampling datasets to {sample_size} samples...")
        train_df = train_df.head(sample_size)
        val_df = val_df.head(sample_size)
        test_df = test_df.head(sample_size)

    # Separate features and target
    # Train
    if TARGET_COL in train_df.columns:
        y_train = train_df[TARGET_COL]
        X_train = train_df.drop(columns=[TARGET_COL])
    else:
        raise ValueError(f"Target column {TARGET_COL} not found in training data")

    # Validation
    if TARGET_COL in val_df.columns:
        y_val = val_df[TARGET_COL]
        X_val = val_df.drop(columns=[TARGET_COL])
    else:
        raise ValueError(f"Target column {TARGET_COL} not found in validation data")

    # Test (Target expected to be missing)
    if TARGET_COL in test_df.columns:
        # In case test set has target (e.g. local testing), drop it
        X_test = test_df.drop(columns=[TARGET_COL])
    else:
        X_test = test_df

    print(
        f"Data loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    return X_train, y_train, X_val, y_val, X_test
