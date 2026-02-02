import os
import pandas as pd
import numpy as np
from library.config import Config


def load_union_dataset():
    """
    Loads the training and validation metadata files and merges them into a single
    union dataset for cross-validation.
    """
    train_path = Config.TRAIN_METADATA_PATH
    val_path = Config.VAL_METADATA_PATH

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train metadata not found at {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Val metadata not found at {val_path}")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    # Merge train and val vertically
    df_union = pd.concat([df_train, df_val], axis=0, ignore_index=True)
    return df_union


def load_test_dataset():
    """
    Loads the test metadata file.
    """
    test_path = Config.TEST_METADATA_PATH
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test metadata not found at {test_path}")
    return pd.read_parquet(test_path)


def clean_data(df):
    """
    Removes potential leakage columns (suffixed with _at_retrieval).
    """
    # Identify leakage columns
    leakage_cols = [col for col in df.columns if col.endswith("_at_retrieval")]

    if leakage_cols:
        df = df.drop(columns=leakage_cols)

    return df


def preprocess_text(df):
    """
    Generates the 'text_combined' column by concatenating title and edit-aware body.
    """
    # Define source columns
    title_col = "request_title"
    body_col = "request_text_edit_aware"

    # Fill NaNs with empty string to ensure successful concatenation
    titles = df[title_col].fillna("").astype(str)
    bodies = df[body_col].fillna("").astype(str)

    # Concatenate with a space separator
    df["text_combined"] = titles + " " + bodies

    return df


def load_and_process_data(load_cached_data=True):
    """
    Main function to load, clean, and preprocess data.
    Implements caching to disk using Parquet format.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        df_train (pd.DataFrame): Processed union training dataset.
        df_test (pd.DataFrame): Processed test dataset.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "processed_data_full.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            df_all = pd.read_parquet(cache_path)

            # Split back into train and test based on the helper column
            df_train = df_all[df_all["is_train"] == 1].copy()
            df_test = df_all[df_all["is_train"] == 0].copy()

            # Clean up helper column
            df_train.drop(columns=["is_train"], inplace=True)
            df_test.drop(columns=["is_train"], inplace=True)

            # Reset indices
            df_train.reset_index(drop=True, inplace=True)
            df_test.reset_index(drop=True, inplace=True)

            return df_train, df_test
        except Exception as e:
            print(f"Error loading cache: {e}. Proceeding to re-process.")

    # 2. Process from scratch
    print("Processing data from scratch...")

    df_train = load_union_dataset()
    df_test = load_test_dataset()

    # Mark splits to allow unified processing
    df_train["is_train"] = 1
    df_test["is_train"] = 0

    # Concatenate for consistent processing
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    # Clean leakage
    df_all = clean_data(df_all)

    # Preprocess text
    df_all = preprocess_text(df_all)

    # Ensure numeric types for metadata columns where possible
    for col in Config.METADATA_COLS:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    # 3. Save to cache
    print(f"Saving processed data to {cache_path}")
    df_all.to_parquet(cache_path, index=False)

    # 4. Return split data
    df_train_processed = df_all[df_all["is_train"] == 1].copy()
    df_test_processed = df_all[df_all["is_train"] == 0].copy()

    df_train_processed.drop(columns=["is_train"], inplace=True)
    df_test_processed.drop(columns=["is_train"], inplace=True)

    # Reset index
    df_train_processed.reset_index(drop=True, inplace=True)
    df_test_processed.reset_index(drop=True, inplace=True)

    return df_train_processed, df_test_processed
