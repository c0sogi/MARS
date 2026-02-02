import os
import pandas as pd
import numpy as np
from library import config


def add_time_weights(df):
    """
    Calculates and appends a time-decay weight column to transactions.

    The weight is calculated as:
        weight = 1 / (days_elapsed + 1) ** DECAY_RATE
    where days_elapsed is the number of days between the transaction date
    and the latest date in the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing a 't_dat' column.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'weight' column.
    """
    # Ensure t_dat is datetime
    if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    # Identify the reference date (the last day in the dataset)
    max_date = df["t_dat"].max()

    # Calculate days elapsed since the reference date
    # (max_date - t_dat) results in a Timedelta series; .dt.days extracts the integer days
    days_elapsed = (max_date - df["t_dat"]).dt.days

    # Apply the time decay formula
    df["weight"] = 1.0 / np.power(days_elapsed + 1, config.DECAY_RATE)

    return df


def load_transactions(load_cached_data=True, use_all_data=True):
    """
    Loads transactions, filters for the most recent weeks, and adds time weights.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from cache.
        use_all_data (bool): If True, loads both training and validation metadata
                             to use the complete history. If False, uses only training data.

    Returns:
        pd.DataFrame: A DataFrame of filtered, weighted transactions.
    """
    # Determine cache filename based on data scope
    cache_filename = (
        "filtered_transactions_all.parquet"
        if use_all_data
        else "filtered_transactions_train.parquet"
    )
    cache_path = config.WORKING_DIR / cache_filename

    # 1. Try Loading from Cache
    if load_cached_data and cache_path.exists():
        print(f"Loading filtered transactions from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Processing transactions from scratch...")

    # 2. Load Raw Data
    # Always load the training set
    df_list = [pd.read_parquet(config.TRAIN_PATH)]

    # Optionally load the validation set (useful for final inference to get full history)
    if use_all_data and config.VAL_PATH.exists():
        df_list.append(pd.read_parquet(config.VAL_PATH))

    df = pd.concat(df_list, ignore_index=True)

    # 3. Preprocessing
    # Convert date column
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    # Filter for the most recent weeks (defined in config)
    max_date = df["t_dat"].max()
    cutoff_date = max_date - pd.Timedelta(weeks=config.WEEKS_TO_KEEP)

    # Keep only recent transactions
    df = df[df["t_dat"] > cutoff_date].copy()

    # Add time-decay weights
    df = add_time_weights(df)

    # 4. Save to Cache
    # Ensure directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Saving filtered transactions to cache: {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


def get_last_purchases(transactions_df, load_cached_data=True):
    """
    Groups data by customer and extracts the most recent article_id for each user.
    This serves as the starting point (source node) for the transition graph predictions.

    Args:
        transactions_df (pd.DataFrame): The transactions DataFrame.
        load_cached_data (bool): If True, attempts to load the result from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'customer_id' and the last purchased 'article_id'.
    """
    cache_path = config.USER_HISTORY_CACHE_PATH

    # 1. Try Loading from Cache
    if load_cached_data and cache_path.exists():
        print(f"Loading user history from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Extracting user history from scratch...")

    # 2. Process Data
    # Ensure date column is datetime for correct sorting
    if not np.issubdtype(transactions_df["t_dat"].dtype, np.datetime64):
        transactions_df = transactions_df.copy()
        transactions_df["t_dat"] = pd.to_datetime(transactions_df["t_dat"])

    # Sort by customer and date to ensure the last row is the most recent purchase
    df_sorted = transactions_df.sort_values(
        ["customer_id", "t_dat"], ascending=[True, True]
    )

    # Group by customer and take the last transaction
    # We only need customer_id and article_id
    last_purchases = df_sorted.groupby("customer_id").tail(1)[
        ["customer_id", "article_id"]
    ]

    # 3. Save to Cache
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Saving user history to cache: {cache_path}")
    last_purchases.to_parquet(cache_path, index=False)

    return last_purchases


def get_user_history_records(transactions_df, load_cached_data=True):
    """
    Retrieves the recent transaction history for users to build the sparse user vector.

    Args:
        transactions_df (pd.DataFrame): Weighted transactions.
        load_cached_data (bool): Whether to use cache.

    Returns:
        pd.DataFrame: DataFrame with 'customer_id', 'article_id', 'weight'.
    """
    cache_path = config.WORKING_DIR / "user_history_records.parquet"

    if load_cached_data and cache_path.exists():
        print(f"Loading user history records from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Extracting user history records from scratch...")

    # Ensure sorted by date (recent first)
    df_sorted = transactions_df.sort_values(
        ["customer_id", "t_dat"], ascending=[True, False]
    )

    # Aggregate weights first (handling multiple purchases of same item)
    grouped = (
        df_sorted.groupby(["customer_id", "article_id"])["weight"].sum().reset_index()
    )

    # Sort by weight descending
    grouped = grouped.sort_values(["customer_id", "weight"], ascending=[True, False])

    # Keep top HISTORY_LENGTH items (Cite solution_lesson_node_00011)
    result = grouped.groupby("customer_id").head(config.HISTORY_LENGTH)

    print(f"Saving user history records to cache: {cache_path}")
    result.to_parquet(cache_path, index=False)

    return result
