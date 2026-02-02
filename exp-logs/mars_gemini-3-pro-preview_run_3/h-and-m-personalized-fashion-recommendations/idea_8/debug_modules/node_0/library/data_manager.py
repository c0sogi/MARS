import pandas as pd
import numpy as np
import os
from library import config


def load_metadata():
    """
    Loads the training, validation, and test metadata from Parquet files.
    Converts the 't_dat' column to datetime objects.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("Loading metadata parquet files...")
    train_df = pd.read_parquet(config.TRAIN_METADATA)
    val_df = pd.read_parquet(config.VAL_METADATA)
    test_df = pd.read_parquet(config.TEST_METADATA)

    # Convert dates
    train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])
    val_df["t_dat"] = pd.to_datetime(val_df["t_dat"])

    # Ensure consistent types
    train_df["article_id"] = train_df["article_id"].astype("int64")
    val_df["article_id"] = val_df["article_id"].astype("int64")

    print(
        f"Metadata loaded. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def get_id_mappings(load_cached_data=True):
    """
    Generates or loads mappings between raw IDs and integer indices.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        tuple: (customer_to_idx, idx_to_customer, article_to_idx, idx_to_article)
               dictionaries and numpy arrays.
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    cust_map_path = config.CUSTOMER_ID_MAP_PATH
    art_map_path = config.ARTICLE_ID_MAP_PATH

    if (
        load_cached_data
        and os.path.exists(cust_map_path)
        and os.path.exists(art_map_path)
    ):
        print("Loading ID mappings from cache...")
        idx_to_customer = np.load(cust_map_path, allow_pickle=True)
        idx_to_article = np.load(art_map_path, allow_pickle=True)
    else:
        print("Computing ID mappings from raw files...")
        # Load raw files to get complete set of IDs
        customers = pd.read_csv(config.CUSTOMERS_CSV)
        articles = pd.read_csv(config.ARTICLES_CSV)

        # Extract unique IDs and sort them to ensure deterministic mapping
        idx_to_customer = np.sort(customers["customer_id"].unique())
        idx_to_article = np.sort(articles["article_id"].unique().astype("int64"))

        # Save to cache
        np.save(cust_map_path, idx_to_customer)
        np.save(art_map_path, idx_to_article)
        print(f"Mappings saved to {config.WORKING_DIR}")

    # Create reverse mappings (Item -> Index)
    # Using a dictionary comprehension is efficient enough for ~1.3M items
    print("Creating reverse mapping dictionaries...")
    customer_to_idx = {cust_id: idx for idx, cust_id in enumerate(idx_to_customer)}
    article_to_idx = {art_id: idx for idx, art_id in enumerate(idx_to_article)}

    print(
        f"Mappings created. Customers: {len(idx_to_customer)}, Articles: {len(idx_to_article)}"
    )
    return customer_to_idx, idx_to_customer, article_to_idx, idx_to_article


def get_sliding_windows():
    """
    Generates temporal splits for the sliding window training strategy.

    Uses config.DATA_END_DATE as the anchor and moves backwards.

    Returns:
        list of dict: Each dict contains 'train_start', 'train_end', 'target_start', 'target_end'.
                      Dates are strings in 'YYYY-MM-DD' format.
    """
    anchor_date = pd.to_datetime(config.DATA_END_DATE)
    windows = []

    print(
        f"Generating {config.RANKER_WINDOW_COUNT} sliding windows ending at {config.DATA_END_DATE}..."
    )

    for i in range(config.RANKER_WINDOW_COUNT):
        # The target week ends at anchor_date - (i * 7 days)
        # i=0: Target ends 2020-09-22
        # i=1: Target ends 2020-09-15

        target_end_dt = anchor_date - pd.Timedelta(days=i * 7)
        target_start_dt = target_end_dt - pd.Timedelta(days=config.TEST_WINDOW_DAYS)

        # History window is strictly before target start
        history_end_dt = target_start_dt
        history_start_dt = history_end_dt - pd.Timedelta(
            weeks=config.HISTORY_WINDOW_SIZE
        )

        # Format as strings (exclusive end date logic often used in pandas slicing,
        # but here we define inclusive ranges for clarity, logic depends on usage)
        # We will use inclusive string comparisons or datetime comparisons downstream.

        window = {
            "history_start": history_start_dt,
            "history_end": history_end_dt,  # Exclusive for history (history < target_start)
            "target_start": target_start_dt,  # Inclusive
            "target_end": target_end_dt,  # Inclusive
        }
        windows.append(window)

        print(
            f"Window {i}: History [{window['history_start'].date()} -> {window['history_end'].date()}) "
            f"| Target [{window['target_start'].date()} -> {window['target_end'].date()}]"
        )

    return windows


def filter_transactions(df, start_date, end_date, inclusive_end=False):
    """
    Helper to filter transactions by date range.

    Args:
        df (pd.DataFrame): DataFrame with 't_dat'.
        start_date (datetime): Start date.
        end_date (datetime): End date.
        inclusive_end (bool): If True, use <= end_date. If False, use < end_date.
    """
    mask = df["t_dat"] >= start_date
    if inclusive_end:
        mask &= df["t_dat"] <= end_date
    else:
        mask &= df["t_dat"] < end_date
    return df.loc[mask].copy()
