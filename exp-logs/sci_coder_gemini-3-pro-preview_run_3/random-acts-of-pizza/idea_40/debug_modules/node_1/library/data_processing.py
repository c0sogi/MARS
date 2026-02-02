import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import Timer


def load_metadata_splits():
    """
    Loads the train, validation, and test metadata from Parquet files.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    train_df = pd.read_parquet(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_parquet(Config.VAL_METADATA_PATH)
    test_df = pd.read_parquet(Config.TEST_METADATA_PATH)
    return train_df, val_df, test_df


def clean_text(df):
    """
    Concatenates request title and edit-aware text into a single column.
    Handles missing values by filling with empty strings.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with 'text_combined' column.
    """
    # Ensure text columns are strings and handle NaNs
    title = df["request_title"].fillna("").astype(str)
    body = df["request_text_edit_aware"].fillna("").astype(str)

    # Concatenate with a space separator
    df["text_combined"] = title + " " + body
    return df


def clean_metadata(train_df, val_df, test_df):
    """
    Selects allowed metadata columns, performs median imputation and standard scaling.
    Fits on train, transforms val and test to prevent leakage.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        val_df (pd.DataFrame): Validation dataframe.
        test_df (pd.DataFrame): Test dataframe.

    Returns:
        tuple: Processed (train_df, val_df, test_df) with scaled dense columns.
    """
    dense_cols = Config.METADATA_DENSE_COLS

    # Extract dense features
    # We verify columns exist to avoid KeyErrors, though Config should match Metadata
    available_cols = [c for c in dense_cols if c in train_df.columns]

    if len(available_cols) != len(dense_cols):
        missing = set(dense_cols) - set(available_cols)
        print(f"Warning: Missing columns in metadata: {missing}")

    X_train = train_df[available_cols].copy()
    X_val = val_df[available_cols].copy()
    X_test = test_df[available_cols].copy()

    # Imputation (Median)
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Update DataFrames with processed values in-place
    train_df[available_cols] = X_train_scaled
    val_df[available_cols] = X_val_scaled
    test_df[available_cols] = X_test_scaled

    return train_df, val_df, test_df


def process_data(load_cached_data=True):
    """
    Main data processing function with caching.
    Loads metadata, cleans text, cleans/scales metadata, and returns processed DataFrames.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    Config.ensure_directories()

    train_cache = os.path.join(Config.CACHE_DIR, "train_base.parquet")
    val_cache = os.path.join(Config.CACHE_DIR, "val_base.parquet")
    test_cache = os.path.join(Config.CACHE_DIR, "test_base.parquet")

    # 1. Try Loading Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print(f"Loading cached processed data from {Config.CACHE_DIR}")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df
        else:
            print("Cache not found or incomplete. Processing from scratch...")

    with Timer("Data Processing"):
        # 2. Load Splits
        train_df, val_df, test_df = load_metadata_splits()

        # 3. Text Cleaning
        train_df = clean_text(train_df)
        val_df = clean_text(val_df)
        test_df = clean_text(test_df)

        # 4. Metadata Cleaning (Imputation + Scaling)
        train_df, val_df, test_df = clean_metadata(train_df, val_df, test_df)

        # 5. Save to Cache
        print(f"Saving processed data to {Config.CACHE_DIR}")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df
