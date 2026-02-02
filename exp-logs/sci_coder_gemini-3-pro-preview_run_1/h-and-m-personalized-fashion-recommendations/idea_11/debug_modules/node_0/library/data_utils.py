import pandas as pd
import numpy as np
import os
from library import config


def load_transactions(path, load_cached_data=True):
    """
    Loads transaction data from a CSV file with caching and type optimization.

    Args:
        path (str): Path to the CSV file (e.g., config.TRAIN_PATH).
        load_cached_data (bool): Whether to attempt loading from the cache.

    Returns:
        pd.DataFrame: The loaded transactions DataFrame.
    """
    # Determine cache path based on input filename
    filename = os.path.basename(path)
    cache_filename = filename.replace(".csv", ".parquet")
    cache_path = os.path.join(config.CACHE_DIR, cache_filename)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached transactions from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Loading transactions from {path}...")
    # Define optimized types
    dtype_dict = {"article_id": "int32", "price": "float32", "sales_channel_id": "int8"}

    # Read CSV
    df = pd.read_csv(path, dtype=dtype_dict)

    # Convert date column
    if "t_dat" in df.columns:
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    # Save to cache
    if load_cached_data:
        print(f"Saving transactions to cache at {cache_path}...")
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


def filter_date_window(df, weeks=config.TRAIN_WEEKS):
    """
    Filters the DataFrame to keep only the last `weeks` of data.

    Args:
        df (pd.DataFrame): The transactions DataFrame containing 't_dat'.
        weeks (int): Number of weeks of history to retain.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if weeks is None or weeks <= 0:
        return df

    max_date = df["t_dat"].max()
    start_date = max_date - pd.Timedelta(weeks=weeks)

    print(
        f"Filtering data from {start_date.date()} to {max_date.date()} ({weeks} weeks)..."
    )
    filtered_df = df[df["t_dat"] >= start_date].copy()

    return filtered_df


def get_time_split(df, val_weeks=config.VAL_WEEKS):
    """
    Splits the DataFrame into training and validation sets based on time.
    The last `val_weeks` are used for validation.

    Args:
        df (pd.DataFrame): The transactions DataFrame.
        val_weeks (int): Number of weeks to use for validation.

    Returns:
        tuple: (train_df, val_df)
    """
    max_date = df["t_dat"].max()
    split_date = max_date - pd.Timedelta(weeks=val_weeks)

    print(f"Splitting data for validation. Split date: {split_date.date()}")

    train_df = df[df["t_dat"] <= split_date].copy()
    val_df = df[df["t_dat"] > split_date].copy()

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    return train_df, val_df


def generate_mappings(transactions_df, customers_df=None, articles_df=None):
    """
    Generates bidirectional mappings for customer_id and article_id.
    Ensures that all customers in customers_df (e.g., test set) are included in the map.

    Args:
        transactions_df (pd.DataFrame): DataFrame containing history.
        customers_df (pd.DataFrame, optional): DataFrame containing all customers (including test).
        articles_df (pd.DataFrame, optional): DataFrame containing all articles.

    Returns:
        tuple: (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
    """
    print("Generating ID mappings...")

    # --- Users ---
    # Collect all unique customers from transactions
    unique_users = set(transactions_df["customer_id"].unique())

    # Add customers from the customers_df (metadata/test.csv or input/customers.csv)
    # to ensure we can predict for cold-start users in the test set
    if customers_df is not None:
        unique_users.update(customers_df["customer_id"].unique())

    sorted_users = sorted(list(unique_users))
    user_to_idx = {u: i for i, u in enumerate(sorted_users)}
    idx_to_user = {i: u for i, u in enumerate(sorted_users)}

    # --- Items ---
    # Collect all unique articles from transactions
    unique_items = set(transactions_df["article_id"].unique())

    # Add articles from articles_df if provided
    if articles_df is not None:
        unique_items.update(articles_df["article_id"].unique())

    sorted_items = sorted(list(unique_items))
    item_to_idx = {i: idx for idx, i in enumerate(sorted_items)}
    idx_to_item = {idx: i for idx, i in enumerate(sorted_items)}

    print(f"Mappings generated. Users: {len(user_to_idx)}, Items: {len(item_to_idx)}")

    return user_to_idx, idx_to_user, item_to_idx, idx_to_item


def load_submission_template():
    """
    Loads the sample submission file to get the list of customers requiring predictions.
    """
    return pd.read_csv(config.SAMPLE_SUBMISSION_PATH)


def load_articles():
    """
    Loads the articles metadata.
    """
    return pd.read_csv(config.ARTICLES_PATH, dtype={"article_id": "int32"})
