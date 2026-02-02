import os
import ast
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, get_common_columns


def load_dataset(load_cached_data=True):
    """
    Loads the dataset for the Pizza Request prediction task.

    This function manages the data pipeline:
    1. Checks for cached processed data (Parquet format) to speed up execution.
    2. If cache is missing or disabled:
       - Loads raw metadata CSVs.
       - Parses stringified list columns (e.g., subreddit history).
       - Aligns columns between Train and Test sets to prevent feature leakage.
       - Imputes missing numerical values using training set statistics.
       - Caches the cleaned data for future runs.

    Args:
        load_cached_data (bool): If True, attempts to load from local Parquet cache.
                                 If False, forces re-processing of raw CSVs.

    Returns:
        tuple: (train_df, val_df, test_df) - The processed Pandas DataFrames.
    """
    set_seed(Config.SEED)

    # Define cache file paths
    train_cache = os.path.join(Config.CACHE_DIR, "train_cleaned.parquet")
    val_cache = os.path.join(Config.CACHE_DIR, "val_cleaned.parquet")
    test_cache = os.path.join(Config.CACHE_DIR, "test_cleaned.parquet")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading cached datasets from Parquet...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df
        else:
            print("Cache not found. Starting data processing...")
    else:
        print("Ignoring cache. Starting data processing...")

    # Load raw CSV data
    print(f"Loading raw data from {Config.INPUT_DIR}...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Handle Debug Mode
    if Config.DEBUG:
        print(
            f"Debug mode enabled: Sampling {Config.DEBUG_SAMPLE_SIZE} rows per split."
        )
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Helper function to parse stringified lists (e.g., "['a', 'b']")
    def parse_list_column(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except (ValueError, SyntaxError):
            return []

    # Parse Subreddit List Column
    print(f"Parsing subreddit history column: {Config.SUBREDDIT_COL}...")
    for df in [train_df, val_df, test_df]:
        if Config.SUBREDDIT_COL in df.columns:
            df[Config.SUBREDDIT_COL] = df[Config.SUBREDDIT_COL].apply(parse_list_column)

    # Align Columns (Leakage Prevention)
    # Identify columns present in both Train and Test to ensure model consistency.
    # We exclude the target column from the intersection check so it's not removed from Train/Val.
    print("Aligning features between Train and Test sets...")
    common_cols = get_common_columns(
        train_df, test_df, exclude_cols=[Config.TARGET_COL]
    )

    # Define final column sets
    # Train/Val must keep the Target. Test only needs features.
    train_cols = common_cols + [Config.TARGET_COL]
    val_cols = common_cols + [Config.TARGET_COL]

    # Ensure ID column is preserved (usually in common_cols, but explicit check for safety)
    if Config.ID_COL not in common_cols:
        common_cols.append(Config.ID_COL)
        train_cols.append(Config.ID_COL)
        val_cols.append(Config.ID_COL)

    # Filter DataFrames
    train_df = train_df[train_cols]
    val_df = val_df[val_cols]
    test_df = test_df[common_cols]

    # Impute Missing Values
    print("Imputing missing values...")

    # 1. Numerical Imputation (Median)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target from imputation
    numeric_cols = [c for c in numeric_cols if c != Config.TARGET_COL]

    # Compute medians on TRAIN set only to avoid data leakage
    medians = train_df[numeric_cols].median()

    train_df[numeric_cols] = train_df[numeric_cols].fillna(medians)
    val_df[numeric_cols] = val_df[numeric_cols].fillna(medians)
    test_df[numeric_cols] = test_df[numeric_cols].fillna(medians)

    # 2. Text Imputation (Empty String)
    if Config.TEXT_COL in train_df.columns:
        train_df[Config.TEXT_COL] = train_df[Config.TEXT_COL].fillna("")
        val_df[Config.TEXT_COL] = val_df[Config.TEXT_COL].fillna("")
        test_df[Config.TEXT_COL] = test_df[Config.TEXT_COL].fillna("")

    # Cache processed data
    print(f"Saving processed data to {Config.CACHE_DIR}...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df
