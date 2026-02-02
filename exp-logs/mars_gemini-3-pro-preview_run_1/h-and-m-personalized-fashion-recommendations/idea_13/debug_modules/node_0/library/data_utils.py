import pandas as pd
import numpy as np
import os
from datetime import timedelta
from library.config import Config


def compute_decay_weights(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Computes exponential decay weights for fast and slow trends based on configuration.
    Adds 'weight_fast' and 'weight_slow' columns to the DataFrame.
    """
    # Ensure t_dat is datetime
    if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    # Calculate days elapsed relative to the last date in the dataset
    max_date = df["t_dat"].max()
    days_elapsed = (max_date - df["t_dat"]).dt.days

    # Calculate decay rates (lambda) from half-lives
    lambda_fast = config.get_decay_rate(config.HALF_LIFE_FAST)
    lambda_slow = config.get_decay_rate(config.HALF_LIFE_SLOW)

    # Compute exponential weights
    # We use float32 to save memory while maintaining sufficient precision
    df["weight_fast"] = np.exp(-lambda_fast * days_elapsed).astype(np.float32)
    df["weight_slow"] = np.exp(-lambda_slow * days_elapsed).astype(np.float32)

    return df


def get_active_inventory(df: pd.DataFrame, config: Config) -> np.ndarray:
    """
    Identifies unique article_ids sold within the active inventory window (last N days).
    Used to mask out obsolete items from the slow graph recommendations.
    """
    if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    max_date = df["t_dat"].max()
    cutoff_date = max_date - timedelta(days=config.ACTIVE_INVENTORY_DAYS)

    # Filter for transactions in the active window
    active_items = df[df["t_dat"] > cutoff_date]["article_id"].unique()
    return active_items


def load_transactions(
    path: str, config: Config, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads transactions with memory optimization, caching, and preprocessing.

    Args:
        path: Path to the raw CSV file.
        config: Configuration object.
        load_cached_data: Whether to attempt loading from the cache.

    Returns:
        Processed DataFrame with correct types and weight columns.
    """
    # Generate a unique cache filename based on the input file
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    cache_path = os.path.join(config.WORKING_DIR, f"{name}_processed.parquet")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Apply debug sampling if enabled
            if config.DEBUG:
                df = df.iloc[-config.DEBUG_SAMPLE_SIZE :].copy()
            return df
        except Exception:
            # Fallback to processing from scratch if cache load fails
            pass

    # 2. Load raw data with optimized types
    # article_id is int32, price is float32, sales_channel is int8
    df = pd.read_csv(
        path,
        dtype={"article_id": "int32", "price": "float32", "sales_channel_id": "int8"},
    )

    # Parse dates
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    # 3. Preprocess: Compute decay weights
    df = compute_decay_weights(df, config)

    # 4. Save to cache (Full dataset)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    # 5. Apply debug sampling if enabled
    if config.DEBUG:
        df = df.iloc[-config.DEBUG_SAMPLE_SIZE :].copy()

    return df


def prepare_time_split(df: pd.DataFrame, config: Config, split_validation: bool = True):
    """
    Splits the DataFrame into history (train) and target (validation) sets based on time.

    Args:
        df: The transaction dataframe.
        config: Configuration object containing window settings.
        split_validation:
            If True, the last 7 days are reserved for validation (target),
            and the preceding HISTORY_WEEKS are used for training.
            If False, the entire dataset (up to HISTORY_WEEKS back) is used for training.

    Returns:
        train_df: Historical data for graph building.
        val_df: Target data for evaluation (empty if split_validation is False).
    """
    if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    max_date = df["t_dat"].max()

    if split_validation:
        # Define validation cutoffs
        # Validation set: Last 7 days
        split_date = max_date - timedelta(days=7)

        val_df = df[df["t_dat"] > split_date].copy()

        # Training set: HISTORY_WEEKS before the split date
        history_start = split_date - timedelta(weeks=config.HISTORY_WEEKS)
        train_df = df[
            (df["t_dat"] <= split_date) & (df["t_dat"] > history_start)
        ].copy()

        return train_df, val_df
    else:
        # Submission mode: Use all data up to max_date
        history_start = max_date - timedelta(weeks=config.HISTORY_WEEKS)
        train_df = df[df["t_dat"] > history_start].copy()

        # Empty validation set
        val_df = pd.DataFrame(columns=df.columns)

        return train_df, val_df
