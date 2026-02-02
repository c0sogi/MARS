import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed


def remove_leakage_features(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Filters the DataFrame to keep only the columns defined in Config.
    This effectively removes leakage features (columns suffixed with _at_retrieval)
    because they are not included in Config.NUMERICAL_COLS.
    """
    # Define the list of allowed columns
    keep_cols = [Config.ID_COL] + Config.TEXT_COLS + Config.NUMERICAL_COLS

    # Include target variable for training/validation sets
    if not is_test:
        keep_cols.append(Config.TARGET_COL)

    # Select only the columns that exist in the dataframe
    # (Intersection of desired columns and existing columns)
    final_cols = [c for c in keep_cols if c in df.columns]

    return df[final_cols].copy()


def load_dataset(
    load_cached_data: bool = True, debug_size: int = Config.DEBUG_SAMPLE_SIZE
):
    """
    Loads, preprocesses, and returns the train, validation, and test datasets.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.
        debug_size (int): If set, subsamples the data for rapid debugging.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    set_seed(Config.RANDOM_SEED)

    # Ensure the working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_train_path = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    cache_val_path = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    cache_test_path = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    # 1. Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):
            print(f"Loading cached datasets from {Config.WORKING_DIR}...")
            train_df = pd.read_parquet(cache_train_path)
            val_df = pd.read_parquet(cache_val_path)
            test_df = pd.read_parquet(cache_test_path)
            return train_df, val_df, test_df

    # 2. Load raw data from metadata
    print("Loading raw datasets from metadata...")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    # 3. Debug Subsampling
    if debug_size is not None:
        print(f"Subsampling datasets to {debug_size} rows for debugging.")
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]

    # 4. Standardize Text Columns
    # The test set only has 'request_text_edit_aware'. The train set has both.
    # To avoid leakage (edits saying "Thanks for pizza") and ensure schema consistency,
    # we map 'request_text_edit_aware' to 'request_text' for all splits.
    print("Standardizing text columns (using edit-aware text)...")
    if "request_text_edit_aware" in train_df.columns:
        train_df["request_text"] = train_df["request_text_edit_aware"]
    if "request_text_edit_aware" in val_df.columns:
        val_df["request_text"] = val_df["request_text_edit_aware"]
    if "request_text_edit_aware" in test_df.columns:
        test_df["request_text"] = test_df["request_text_edit_aware"]

    # 4b. Feature Extraction (Derived Features)
    print("Extracting derived features (temporal & text length)...")
    for df in [train_df, val_df, test_df]:
        # Temporal features from timestamp
        if "unix_timestamp_of_request" in df.columns:
            dt = pd.to_datetime(df["unix_timestamp_of_request"], unit="s")
            df["request_hour"] = dt.dt.hour
            df["request_day"] = dt.dt.dayofweek

        # Text length features
        # Ensure text is string and fillna
        texts = df["request_text"].fillna("").astype(str)
        df["request_text_len"] = texts.str.len()
        df["request_word_count"] = texts.apply(lambda x: len(x.split()))

    # 5. Feature Selection / Leakage Removal
    print("Selecting features and removing leakage...")
    train_df = remove_leakage_features(train_df, is_test=False)
    val_df = remove_leakage_features(val_df, is_test=False)
    test_df = remove_leakage_features(test_df, is_test=True)

    # 6. Impute Numerical Values
    print("Imputing missing numerical values...")
    # Identify numerical columns that are actually present
    num_cols = [c for c in Config.NUMERICAL_COLS if c in train_df.columns]

    if num_cols:
        # Compute medians on the training set only
        medians = train_df[num_cols].median()

        # Apply medians to all splits
        train_df[num_cols] = train_df[num_cols].fillna(medians)
        val_df[num_cols] = val_df[num_cols].fillna(medians)
        test_df[num_cols] = test_df[num_cols].fillna(medians)

    # 7. Text Concatenation
    # Create 'combined_text' for the Bag-of-Words model
    print("Creating 'combined_text' feature...")
    for df in [train_df, val_df, test_df]:
        # Fill NaNs with empty strings to allow concatenation
        title = df["request_title"].fillna("").astype(str)
        text = df["request_text"].fillna("").astype(str)
        df["combined_text"] = title + " " + text

    # 8. Save to Cache
    print(f"Saving processed datasets to {Config.WORKING_DIR}...")
    train_df.to_parquet(cache_train_path, index=False)
    val_df.to_parquet(cache_val_path, index=False)
    test_df.to_parquet(cache_test_path, index=False)

    return train_df, val_df, test_df
