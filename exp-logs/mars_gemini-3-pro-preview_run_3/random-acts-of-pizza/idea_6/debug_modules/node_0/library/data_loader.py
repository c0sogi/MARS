import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_TRAIN_PROCESSED,
    CACHE_VAL_PROCESSED,
    CACHE_TEST_PROCESSED,
    DROP_COLS,
    TARGET_COL,
    TEXT_COL,
    TITLE_COL,
    SUBREDDIT_COL,
    ID_COL,
    SEED,
    DEBUG_SAMPLE_SIZE,
    WORKING_DIR,
)
from library.utils import timer, set_seed


def clean_data(
    df: pd.DataFrame, is_train: bool = False, train_medians: pd.Series = None
) -> pd.DataFrame:
    """
    Performs data cleaning, leakage prevention, and imputation.

    Args:
        df: Input DataFrame.
        is_train: Boolean indicating if this is the training set (to calculate medians).
        train_medians: Series of medians from the training set (for val/test imputation).

    Returns:
        Cleaned DataFrame and calculated medians (if is_train=True).
    """
    # 1. Leakage Prevention: Drop columns ending with '_at_retrieval'
    retrieval_cols = [c for c in df.columns if c.endswith("_at_retrieval")]
    df = df.drop(columns=retrieval_cols, errors="ignore")

    # 2. Drop configured DROP_COLS
    # CRITICAL: We must retain the columns needed for feature engineering (Text, Title, Subreddits)
    # even if they are listed in DROP_COLS in the config (which implies they are dropped from *tabular* view).
    cols_to_keep = {TEXT_COL, TITLE_COL, SUBREDDIT_COL, ID_COL}
    if is_train:
        cols_to_keep.add(TARGET_COL)

    # Filter DROP_COLS to exclude essential feature columns
    final_drop_cols = [c for c in DROP_COLS if c not in cols_to_keep]
    df = df.drop(columns=final_drop_cols, errors="ignore")

    # 3. Imputation
    # Identify numerical columns (excluding target and ID)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    if is_train:
        # Calculate medians for training set
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)
        return df, medians
    else:
        # Apply training medians to val/test
        if train_medians is not None:
            # Only impute columns that exist in both (intersection)
            common_cols = [c for c in numeric_cols if c in train_medians.index]
            df[common_cols] = df[common_cols].fillna(train_medians[common_cols])
        return df


def load_data(load_cached_data: bool = True, debug: bool = False):
    """
    Loads, cleans, and returns the training, validation, and test datasets.

    Args:
        load_cached_data: If True, attempts to load processed parquet files from cache.
        debug: If True, downsamples the dataset for rapid prototyping.

    Returns:
        Tuple containing:
        - X_train (pd.DataFrame): Training features
        - y_train (pd.Series): Training target
        - X_val (pd.DataFrame): Validation features
        - y_val (pd.Series): Validation target
        - X_test (pd.DataFrame): Test features
        - test_ids (np.ndarray): Request IDs for the test set
    """
    set_seed(SEED)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Check if cache exists
    cache_exists = (
        os.path.exists(CACHE_TRAIN_PROCESSED)
        and os.path.exists(CACHE_VAL_PROCESSED)
        and os.path.exists(CACHE_TEST_PROCESSED)
    )

    if load_cached_data and cache_exists and not debug:
        print("Loading processed data from cache...")
        with timer("Load Cache"):
            train_df = pd.read_parquet(CACHE_TRAIN_PROCESSED)
            val_df = pd.read_parquet(CACHE_VAL_PROCESSED)
            test_df = pd.read_parquet(CACHE_TEST_PROCESSED)
    else:
        print("Loading raw data from metadata...")
        with timer("Load Raw"):
            train_df = pd.read_parquet(TRAIN_PATH)
            val_df = pd.read_parquet(VAL_PATH)
            test_df = pd.read_parquet(TEST_PATH)

        if debug:
            print(f"DEBUG MODE: Sampling {DEBUG_SAMPLE_SIZE} rows.")
            train_df = train_df.head(DEBUG_SAMPLE_SIZE).copy()
            val_df = val_df.head(DEBUG_SAMPLE_SIZE).copy()
            test_df = test_df.head(DEBUG_SAMPLE_SIZE).copy()

        print("Cleaning and imputing data...")
        with timer("Preprocessing"):
            # Clean Train and get medians
            train_df, medians = clean_data(train_df, is_train=True)

            # Clean Val and Test using Train medians
            val_df = clean_data(val_df, is_train=False, train_medians=medians)
            test_df = clean_data(test_df, is_train=False, train_medians=medians)

        # Save to cache (only if not debugging, to avoid overwriting full cache with debug data)
        if not debug:
            print("Saving processed data to cache...")
            train_df.to_parquet(CACHE_TRAIN_PROCESSED, index=False)
            val_df.to_parquet(CACHE_VAL_PROCESSED, index=False)
            test_df.to_parquet(CACHE_TEST_PROCESSED, index=False)

    # Separate Features and Target
    print("Preparing final datasets...")

    # Train
    y_train = train_df[TARGET_COL]
    X_train = train_df.drop(columns=[TARGET_COL])

    # Val
    y_val = val_df[TARGET_COL]
    X_val = val_df.drop(columns=[TARGET_COL])

    # Test (Extract IDs first)
    test_ids = test_df[ID_COL].values
    X_test = test_df  # Keep ID in X_test for now if needed, or drop?
    # Usually X_test should match X_train columns.
    # X_train still has ID_COL because we kept it in cols_to_keep.
    # We will keep ID_COL in the DataFrames for alignment during feature engineering,
    # but the models will likely drop it.

    print(
        f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )

    return X_train, y_train, X_val, y_val, X_test, test_ids
