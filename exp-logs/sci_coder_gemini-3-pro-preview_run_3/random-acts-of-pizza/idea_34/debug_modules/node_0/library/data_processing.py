import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from library import config
from library import utils


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs text cleaning and constructs the holistic text feature.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: Dataframe with cleaned text and 'holistic_text' column.
    """
    df = df.copy()

    # Fill NaNs in standard text columns with empty strings
    for col in config.TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)

    # Process subreddits list into a space-separated string
    # The raw data might have this as a list, numpy array, or None
    def process_subreddits(x):
        if x is None:
            return ""
        if isinstance(x, (list, np.ndarray)):
            # Filter out any non-string elements just in case, though unlikely in this dataset
            return " ".join([str(s) for s in x if s])
        return str(x)

    subreddit_str = df[config.SUBREDDIT_COL].apply(process_subreddits)

    # Construct Holistic Text: Title + Body + Subreddit History
    # This enables the Interaction Bagger to find patterns across these modalities
    df["holistic_text"] = (
        df["request_title"] + " " + df["request_text_edit_aware"] + " " + subreddit_str
    )

    return df


def process_metadata(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
):
    """
    Imputes and scales the allow-listed metadata columns.
    Fits transformers on Train only to prevent leakage.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.

    Returns:
        tuple: Processed (train_df, val_df, test_df).
    """
    meta_cols = config.METADATA_COLS

    # Extract values
    X_train = train_df[meta_cols].values
    X_val = val_df[meta_cols].values
    X_test = test_df[meta_cols].values

    # 1. Imputation (Median)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # 2. Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Assign back to dataframes (using copy to ensure safety)
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    train_out[meta_cols] = X_train_scaled
    val_out[meta_cols] = X_val_scaled
    test_out[meta_cols] = X_test_scaled

    return train_out, val_out, test_out


def load_and_process_data(load_cached_data: bool = True):
    """
    Main entry point for data processing.
    Loads raw data, cleans text, processes metadata, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df) processed dataframes.
    """
    # Define cache file paths
    train_cache_path = os.path.join(config.CACHE_DIR, "train_processed.parquet")
    val_cache_path = os.path.join(config.CACHE_DIR, "val_processed.parquet")
    test_cache_path = os.path.join(config.CACHE_DIR, "test_processed.parquet")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            print("Loading processed data from cache...")
            try:
                train_df = utils.load_parquet(train_cache_path)
                val_df = utils.load_parquet(val_cache_path)
                test_df = utils.load_parquet(test_cache_path)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Error loading cache: {e}. Reprocessing from scratch.")
        else:
            print("Cache not found. Processing from scratch...")

    # 2. Load Raw Data
    print("Loading raw metadata...")
    train_df = utils.load_parquet(config.TRAIN_PATH)
    val_df = utils.load_parquet(config.VAL_PATH)
    test_df = utils.load_parquet(config.TEST_PATH)

    # 3. Text Processing
    print("Constructing holistic text features...")
    train_df = clean_text_fields(train_df)
    val_df = clean_text_fields(val_df)
    test_df = clean_text_fields(test_df)

    # 4. Metadata Processing
    print("Processing metadata (Imputation & Scaling)...")
    train_df, val_df, test_df = process_metadata(train_df, val_df, test_df)

    # 5. Column Selection & Leakage Prevention
    # We strictly keep only the columns needed for the pipeline
    # ID, Text Cols, Subreddit Col, Holistic Col, Metadata Cols, Target (if present)

    base_cols = (
        [config.ID_COL]
        + config.TEXT_COLS
        + [config.SUBREDDIT_COL, "holistic_text"]
        + config.METADATA_COLS
    )

    train_cols = base_cols + [config.TARGET_COL]
    val_cols = base_cols + [config.TARGET_COL]
    test_cols = base_cols  # Test set does not have the target

    train_df = train_df[train_cols]
    val_df = val_df[val_cols]
    test_df = test_df[test_cols]

    # 6. Save to Cache
    print(f"Saving processed data to {config.CACHE_DIR}...")
    utils.save_parquet(train_df, train_cache_path)
    utils.save_parquet(val_df, val_cache_path)
    utils.save_parquet(test_df, test_cache_path)

    return train_df, val_df, test_df
