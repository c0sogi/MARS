import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    LEAKAGE_COLUMNS,
    TEXT_COLUMN,
)


def engineer_features(df):
    """
    Performs feature engineering on the dataframe.
    - Fills missing text and creates length-based meta-features.
    - Encodes specific categorical columns (e.g., user flair).
    - Converts boolean columns to integers.
    - Computes ratio-based features from upvotes/downvotes.
    - Imputes missing numerical values with 0.
    """
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Text Handling
    # Ensure text column is string and fill NaNs
    if TEXT_COLUMN in df.columns:
        df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

        # Meta-features: Character count and Word count
        # Log-transform to handle skew and help linear models
        df["text_len_chars"] = np.log1p(df[TEXT_COLUMN].apply(len))
        df["text_len_words"] = np.log1p(df[TEXT_COLUMN].apply(lambda x: len(x.split())))

    # 2. Categorical Handling
    # Map 'requester_user_flair' to numeric values
    if "requester_user_flair" in df.columns:
        flair_map = {None: 0, np.nan: 0, "None": 0, "shroom": 1, "PIF": 2}
        # Map, fill any remaining NaNs (e.g. from new categories) with 0, and cast to int
        df["requester_user_flair"] = (
            df["requester_user_flair"].map(flair_map).fillna(0).astype(int)
        )

    # 3. Boolean Handling
    # Convert all boolean columns to integers (0/1)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # 4. Ratio Features
    # Create upvote ratio from plus/minus columns if they exist
    plus_col = "requester_upvotes_plus_downvotes_at_request"
    minus_col = "requester_upvotes_minus_downvotes_at_request"

    if plus_col in df.columns and minus_col in df.columns:
        # Fill NaNs before calculation to ensure safety
        plus = df[plus_col].fillna(0)
        minus = df[minus_col].fillna(0)

        # Calculate derived upvotes count: (Total + Net) / 2
        upvotes = (plus + minus) / 2

        # Calculate ratio: Upvotes / Total (handle division by zero)
        # If total is 0, we assign a neutral ratio of 0.5
        df["requester_upvote_ratio_at_request"] = np.where(
            plus > 0, upvotes / plus, 0.5
        )

    # 5. Numerical Imputation
    # Fill remaining numerical NaNs with 0 (suitable for count data and safe for RF/LR pipelines)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def get_common_features(train_df, test_df):
    """
    Identifies common numerical features between train and test sets,
    excluding defined leakage columns.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # 1. Intersection: Only keep columns present in both datasets
    common_cols = train_cols.intersection(test_cols)

    # 2. Leakage Exclusion: Remove columns defined in LEAKAGE_COLUMNS
    features = [c for c in common_cols if c not in LEAKAGE_COLUMNS]

    # 3. Type Filter: Keep only numeric columns
    # This filters out raw text, lists, or other objects that haven't been processed
    valid_features = []
    for c in features:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            valid_features.append(c)

    return sorted(valid_features)


def load_data(load_cached_data=True):
    """
    Loads, processes, and caches the data.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        train_df (pd.DataFrame): Processed training data.
        val_df (pd.DataFrame): Processed validation data.
        test_df (pd.DataFrame): Processed test data.
        feature_cols (list): List of safe, common numerical feature names.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_cache_path = os.path.join(CACHE_DIR, "train_processed.parquet")
    val_cache_path = os.path.join(CACHE_DIR, "val_processed.parquet")
    test_cache_path = os.path.join(CACHE_DIR, "test_processed.parquet")

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {CACHE_DIR}...")
        train_df = pd.read_parquet(train_cache_path)
        val_df = pd.read_parquet(val_cache_path)
        test_df = pd.read_parquet(test_cache_path)
    else:
        print("Processing data from scratch...")
        # Load raw metadata CSVs
        train_df = pd.read_csv(TRAIN_PATH)
        val_df = pd.read_csv(VAL_PATH)
        test_df = pd.read_csv(TEST_PATH)

        # Apply feature engineering
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

        # Save processed data to cache
        print(f"Saving processed data to {CACHE_DIR}...")
        train_df.to_parquet(train_cache_path, index=False)
        val_df.to_parquet(val_cache_path, index=False)
        test_df.to_parquet(test_cache_path, index=False)

    # Identify the list of features to be used by the models
    feature_cols = get_common_features(train_df, test_df)

    print(
        f"Data Loaded. Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )
    print(f"Number of common numerical features: {len(feature_cols)}")

    return train_df, val_df, test_df, feature_cols
