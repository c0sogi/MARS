import pandas as pd
import numpy as np
import os
from library.config import Config


def get_temporal_view(
    df: pd.DataFrame, days_back: int, reference_date: str = Config.REFERENCE_DATE
) -> pd.DataFrame:
    """
    Extracts a temporal slice of the transactions DataFrame based on a lookback window.

    This function is critical for the 'Dual-Window' architecture, allowing the extraction
    of different time periods for Structure Learning (Long-term), Intent Inference (Short-term),
    and Inventory Feasibility (Immediate).

    Args:
        df: The dataframe containing transactions with a 't_dat' column.
        days_back: Number of days to look back from the reference date.
        reference_date: The anchor date (string 'YYYY-MM-DD').

    Returns:
        pd.DataFrame: A subset of df containing transactions within (reference_date - days_back, reference_date].
    """
    ref_date = pd.to_datetime(reference_date)
    start_date = ref_date - pd.Timedelta(days=days_back)

    # Filter strictly greater than start_date and less than or equal to ref_date
    # Example: Ref=2020-09-22, Days=7 -> Start=2020-09-15
    # Window includes: 16, 17, 18, 19, 20, 21, 22 (7 days)
    mask = (df["t_dat"] > start_date) & (df["t_dat"] <= ref_date)
    return df.loc[mask].copy()


def load_processed_data(load_cached_data: bool = True):
    """
    Loads transaction data, generates compact integer mappings for users and items,
    and returns processed dataframes. Implements strict caching logic using Parquet.

    Args:
        load_cached_data: If True, attempts to load pre-computed files from the working directory.

    Returns:
        tuple: (transactions_df, user_map, item_map)
            - transactions_df: DataFrame with ['t_dat', 'user_idx', 'item_idx', 'price', 'sales_channel_id']
            - user_map: DataFrame mapping 'customer_id' to 'user_idx'
            - item_map: DataFrame mapping 'article_id' to 'item_idx'
    """
    # Ensure working directory exists as per requirements
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    cache_trans = os.path.join(Config.WORKING_DIR, "transactions_processed.parquet")
    cache_user = Config.CACHE_USER_MAP
    cache_item = Config.CACHE_ITEM_MAP

    # Check if all required cache files exist
    cache_exists = (
        os.path.exists(cache_trans)
        and os.path.exists(cache_user)
        and os.path.exists(cache_item)
    )

    # Logic: Load from cache if requested and available
    if load_cached_data and cache_exists:
        print(f"Loading processed data from cache: {Config.WORKING_DIR}")
        transactions = pd.read_parquet(cache_trans)
        user_map = pd.read_parquet(cache_user)
        item_map = pd.read_parquet(cache_item)
        return transactions, user_map, item_map

    print("Cache not found or ignored. Processing data from scratch...")

    # --- 1. Load and Unify Raw Data ---
    # Load training and validation metadata to form the complete history
    print("Loading raw metadata CSVs...")
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # Concatenate to get full transaction history
    transactions = pd.concat([df_train, df_val], ignore_index=True)

    # Convert dates to datetime objects for temporal slicing
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])

    # --- 2. Generate Integer Mappings ---
    # User Map: Must include all historical customers AND all test customers to ensure
    # we can generate predictions for everyone required in the submission.
    df_test = pd.read_csv(Config.TEST_CSV)

    unique_customers = pd.concat(
        [transactions["customer_id"], df_test["customer_id"]]
    ).unique()

    user_map = pd.DataFrame({"customer_id": unique_customers})
    user_map["user_idx"] = np.arange(len(user_map), dtype=np.int32)

    # Item Map: Must include all historical articles AND all available articles
    # to maintain consistent matrix dimensions.
    df_articles = pd.read_csv(Config.ARTICLES_CSV)

    unique_articles = pd.concat(
        [transactions["article_id"], df_articles["article_id"]]
    ).unique()

    item_map = pd.DataFrame({"article_id": unique_articles})
    item_map["item_idx"] = np.arange(len(item_map), dtype=np.int32)

    # --- 3. Map Transactions to Integers ---
    print("Mapping IDs to integers...")
    # Merge to map string/large-int IDs to compact int32 indices
    transactions = transactions.merge(user_map, on="customer_id", how="left")
    transactions = transactions.merge(item_map, on="article_id", how="left")

    # Select relevant columns and optimize data types
    cols = ["t_dat", "user_idx", "item_idx", "price", "sales_channel_id"]
    transactions = transactions[cols]

    # Enforce strict types for memory efficiency
    transactions["user_idx"] = transactions["user_idx"].astype(np.int32)
    transactions["item_idx"] = transactions["item_idx"].astype(np.int32)
    transactions["sales_channel_id"] = transactions["sales_channel_id"].astype(np.int8)
    transactions["price"] = transactions["price"].astype(np.float32)

    # --- 4. Save to Cache ---
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    transactions.to_parquet(cache_trans, index=False)
    user_map.to_parquet(cache_user, index=False)
    item_map.to_parquet(cache_item, index=False)

    return transactions, user_map, item_map
