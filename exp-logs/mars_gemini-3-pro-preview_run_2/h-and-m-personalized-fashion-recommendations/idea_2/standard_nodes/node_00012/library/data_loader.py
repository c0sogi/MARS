import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    WORKING_DIR,
    RETRIEVAL_HISTORY_WEEKS,
    VAL_DAYS,
    DATE_COL,
)


def load_metadata():
    """
    Loads the raw metadata parquet files.

    Returns
    -------
    tuple
        (df_train_meta, df_val_meta, df_test_meta)
    """
    # Using pandas read_parquet for efficiency
    df_train = pd.read_parquet(TRAIN_META_PATH)
    df_val = pd.read_parquet(VAL_META_PATH)
    df_test = pd.read_parquet(TEST_META_PATH)
    return df_train, df_val, df_test


def get_time_split_data(
    load_cached_data=True, val_days=VAL_DAYS, train_weeks=RETRIEVAL_HISTORY_WEEKS
):
    """
    Generates time-based split data for Stage 1 (Retrieval) and Stage 2 (Ranking).

    Strategy:
    - Validation Set: Last `val_days` of data.
    - Training Set: `train_weeks` immediately preceding the validation period.

    Parameters
    ----------
    load_cached_data : bool
        If True, attempts to load pre-computed splits from disk.
    val_days : int
        Number of days for the validation set.
    train_weeks : int
        Number of weeks of history to use for the training set.

    Returns
    -------
    tuple
        (train_df, val_df)
    """
    # Define cache paths
    train_cache_path = WORKING_DIR / "time_split_train.parquet"
    val_cache_path = WORKING_DIR / "time_split_val.parquet"

    # 1. Try loading from cache
    if load_cached_data:
        if train_cache_path.exists() and val_cache_path.exists():
            print(f"Loading cached time-split data from {WORKING_DIR}...")
            train_df = pd.read_parquet(train_cache_path)
            val_df = pd.read_parquet(val_cache_path)
            return train_df, val_df
        else:
            print("Cache not found. Computing splits from scratch...")

    # 2. Load and Merge Data
    # We need the full history to perform a global time split, so we combine the metadata train/val splits.
    # The metadata split was customer-based, but here we need a time-based view.
    print("Loading metadata...")
    df_meta_train, df_meta_val, _ = load_metadata()

    # Concatenate to get full transaction history
    full_df = pd.concat([df_meta_train, df_meta_val], axis=0, ignore_index=True)

    # Ensure date column is datetime
    full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL])

    # 3. Calculate Split Dates
    max_date = full_df[DATE_COL].max()
    split_date = max_date - pd.Timedelta(days=val_days)
    start_date = split_date - pd.Timedelta(weeks=train_weeks)

    print(f"Max Date in Data: {max_date.date()}")
    print(
        f"Validation Period: {(split_date + pd.Timedelta(days=1)).date()} to {max_date.date()}"
    )
    print(
        f"Training Period: {(start_date + pd.Timedelta(days=1)).date()} to {split_date.date()}"
    )

    # 4. Filter Data
    # Validation: strictly greater than split_date
    val_mask = full_df[DATE_COL] > split_date
    val_df = full_df[val_mask].copy()

    # Training: between start_date and split_date (inclusive of split_date)
    train_mask = (full_df[DATE_COL] > start_date) & (full_df[DATE_COL] <= split_date)
    train_df = full_df[train_mask].copy()

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # 5. Save to Cache
    print(f"Saving splits to {WORKING_DIR}...")
    os.makedirs(WORKING_DIR, exist_ok=True)
    train_df.to_parquet(train_cache_path, index=False)
    val_df.to_parquet(val_cache_path, index=False)

    return train_df, val_df


def load_test_customers():
    """
    Loads the test dataset which contains the customers to predict for.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 'customer_id' column.
    """
    return pd.read_parquet(TEST_META_PATH)
