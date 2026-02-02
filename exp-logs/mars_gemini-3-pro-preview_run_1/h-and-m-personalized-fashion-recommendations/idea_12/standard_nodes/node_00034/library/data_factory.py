import pandas as pd
import numpy as np
import os
import hashlib
from datetime import timedelta
from library import config


def load_and_preprocess(file_path=config.TRAIN_META_PATH, load_cached_data=True):
    """
    Loads transaction data, preprocesses dates, computes days_elapsed, and handles caching.

    Args:
        file_path (str): Path to the CSV file containing transactions.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Preprocessed transactions dataframe.
    """
    # Generate a unique cache filename based on the input file path hash
    # This ensures different input files (train vs val vs full) get their own cache
    file_hash = hashlib.md5(os.path.abspath(file_path).encode("utf-8")).hexdigest()
    filename = os.path.basename(file_path).replace(".csv", "")
    cache_path = os.path.join(
        config.WORKING_DIR, f"processed_{filename}_{file_hash}.parquet"
    )

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Process from Scratch
    print(f"Processing data from {file_path}...")

    # Load raw CSV with optimized types to save memory
    # Using config.DTYPE_OPTS ensures article_id is int32, etc.
    df = pd.read_csv(file_path, dtype=config.DTYPE_OPTS)

    # Convert t_dat to datetime objects
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    # Compute days_elapsed relative to the dataset's absolute max date
    # This provides a global reference. Specific splits will adjust this relative to their split date.
    max_date = df["t_dat"].max()
    # Result is timedelta, convert to days
    df["days_elapsed"] = (max_date - df["t_dat"]).dt.days.astype("int16")

    # 3. Save to Cache
    # Ensure the working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    print(f"Saving processed data to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


def get_time_split(df, val_days=7, train_weeks=config.TRAIN_HISTORY_WEEKS):
    """
    Splits the dataframe into training and validation sets based on time.

    The split is defined as:
    - Validation: The last 'val_days' (Week T).
    - Training: The 'train_weeks' immediately preceding the validation period (Weeks T-20 to T-1).

    Args:
        df (pd.DataFrame): Preprocessed dataframe with 't_dat' and 'days_elapsed'.
        val_days (int): Number of days in the validation set.
        train_weeks (int): Number of weeks of history to include in the training set.

    Returns:
        tuple: (train_df, val_df)
    """
    print(f"Splitting data: Val days={val_days}, Train weeks={train_weeks}")

    # Determine split points
    max_date = df["t_dat"].max()
    split_date = max_date - timedelta(days=val_days)
    train_start_date = split_date - timedelta(weeks=train_weeks)

    # Create temporal masks
    # Validation: strictly after split_date
    val_mask = df["t_dat"] > split_date
    # Training: up to split_date, and within the history window
    train_mask = (df["t_dat"] <= split_date) & (df["t_dat"] > train_start_date)

    # Create copies to avoid SettingWithCopy warnings and to allow modification
    val_df = df.loc[val_mask].copy()
    train_df = df.loc[train_mask].copy()

    # Adjust days_elapsed for the training set
    # The model expects 'days_elapsed=0' to represent the day before prediction starts.
    # In the raw df, 'days_elapsed=0' is the end of the validation period.
    # For the training set, the "current" time is 'split_date'.
    # Therefore, we shift days_elapsed by subtracting 'val_days'.
    train_df["days_elapsed"] = train_df["days_elapsed"] - val_days

    # Sanity check: ensure no negative days_elapsed in training (should be impossible by definition of mask)
    if not train_df.empty:
        min_elapsed = train_df["days_elapsed"].min()
        if min_elapsed < 0:
            # Filter out any data that falls into the future relative to split_date
            # (This is a safety fallback, though mask logic should prevent it)
            train_df = train_df[train_df["days_elapsed"] >= 0]

    print(
        f"Train set: {len(train_df)} rows ({train_df['t_dat'].min().date()} to {train_df['t_dat'].max().date()})"
    )
    print(
        f"Val set: {len(val_df)} rows ({val_df['t_dat'].min().date()} to {val_df['t_dat'].max().date()})"
    )

    return train_df, val_df
