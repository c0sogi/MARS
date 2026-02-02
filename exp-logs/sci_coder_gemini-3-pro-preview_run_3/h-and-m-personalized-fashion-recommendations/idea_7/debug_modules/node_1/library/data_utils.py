import os
import pandas as pd
import numpy as np
from datetime import timedelta
from library.config import Config


def load_dataset():
    """
    Loads the raw datasets from the metadata and input directories.
    Converts date columns to datetime objects.

    Returns:
        train_df (pd.DataFrame): Training transactions.
        val_df (pd.DataFrame): Validation transactions.
        test_df (pd.DataFrame): Test set customers (sample submission).
        articles_df (pd.DataFrame): Article metadata.
        customers_df (pd.DataFrame): Customer metadata.
    """
    # Load Parquet files (Metadata)
    train_df = pd.read_parquet(Config.TRAIN_DATA_PATH)
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)
    test_df = pd.read_parquet(Config.TEST_DATA_PATH)

    # Load CSV files (Raw Input)
    articles_df = pd.read_csv(Config.ARTICLES_PATH)
    customers_df = pd.read_csv(Config.CUSTOMERS_PATH)

    # Convert dates to datetime
    train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])
    val_df["t_dat"] = pd.to_datetime(val_df["t_dat"])

    return train_df, val_df, test_df, articles_df, customers_df


def get_id_maps(load_cached_data=True):
    """
    Generates or loads the mapping between raw IDs and integer indices.
    Ensures consistent mapping for sparse matrices.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        customer_to_idx (dict): Map raw customer_id -> int.
        article_to_idx (dict): Map raw article_id -> int.
        customer_id_map (np.array): Array where index i -> raw customer_id.
        article_id_map (np.array): Array where index i -> raw article_id.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_exists = (
        Config.CACHE_ARTICLE_ID_MAP.exists() and Config.CACHE_CUSTOMER_ID_MAP.exists()
    )

    if load_cached_data and cache_exists:
        print("Loading ID maps from cache...")
        article_id_map = np.load(Config.CACHE_ARTICLE_ID_MAP, allow_pickle=True)
        customer_id_map = np.load(Config.CACHE_CUSTOMER_ID_MAP, allow_pickle=True)
    else:
        print("Computing ID maps from scratch...")
        # Load data to find all unique IDs
        train_df, val_df, test_df, articles_df, customers_df = load_dataset()

        # Articles: Use all available articles
        # We sort to ensure determinism
        article_id_map = articles_df["article_id"].unique()
        article_id_map.sort()

        # Customers: Union of all known customers (metadata + raw list)
        # This ensures we don't crash on cold-start users in test set
        cust_ids_train = train_df["customer_id"].unique()
        cust_ids_val = val_df["customer_id"].unique()
        cust_ids_test = test_df["customer_id"].unique()
        cust_ids_raw = customers_df["customer_id"].unique()

        unique_customers = np.unique(
            np.concatenate([cust_ids_train, cust_ids_val, cust_ids_test, cust_ids_raw])
        )
        customer_id_map = unique_customers

        # Save to cache
        np.save(Config.CACHE_ARTICLE_ID_MAP, article_id_map)
        np.save(Config.CACHE_CUSTOMER_ID_MAP, customer_id_map)
        print(f"Saved ID maps to {Config.WORKING_DIR}")

    # Build reverse mappings (Dicts for fast lookup)
    print("Building reverse ID lookup dictionaries...")
    article_to_idx = {aid: i for i, aid in enumerate(article_id_map)}
    customer_to_idx = {cid: i for i, cid in enumerate(customer_id_map)}

    return customer_to_idx, article_to_idx, customer_id_map, article_id_map


def filter_by_date(df, start_date, end_date):
    """
    Filters a dataframe by a date range [start_date, end_date].
    """
    mask = (df["t_dat"] >= start_date) & (df["t_dat"] <= end_date)
    return df.loc[mask].copy()


def get_sliding_windows(max_date, num_windows=5):
    """
    Generates date ranges for the sliding window training strategy.

    Strategy:
    - We want to train a ranker to predict purchases in a specific 'target week'.
    - The features for the ranker are generated from the 'history period' immediately preceding the target week.
    - Config.RANKER_WINDOW_WEEKS defines the total span (History + Target).
    - We assume the Target Period is always 1 week (7 days).
    - History Period = RANKER_WINDOW_WEEKS - 1 weeks.

    Args:
        max_date (pd.Timestamp): The last available date in the training data.
        num_windows (int): Number of sliding windows to generate.

    Returns:
        windows (list of tuples): Each tuple is (hist_start, hist_end, target_start, target_end).
    """
    windows = []
    current_target_end = max_date

    # Total span in days
    total_days = Config.RANKER_WINDOW_WEEKS * 7
    target_days = 7
    history_days = total_days - target_days

    for _ in range(num_windows):
        # Target Week
        target_end = current_target_end
        target_start = target_end - timedelta(days=target_days - 1)

        # History Period (immediately before target)
        hist_end = target_start - timedelta(days=1)
        hist_start = hist_end - timedelta(days=history_days - 1)

        windows.append((hist_start, hist_end, target_start, target_end))

        # Slide back by 1 week for the next window
        current_target_end = current_target_end - timedelta(weeks=1)

    return windows
