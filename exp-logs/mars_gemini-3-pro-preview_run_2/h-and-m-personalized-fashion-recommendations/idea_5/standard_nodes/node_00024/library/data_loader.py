import pandas as pd
import numpy as np
import torch
from pathlib import Path
from library import config
from library import utils


def load_raw_data():
    """
    Loads and merges training and validation metadata to form the full history.
    Also loads articles and customers metadata.

    Returns:
        tuple: (df_history, df_articles, df_customers)
    """
    print("Loading raw data from metadata...")

    # Load history
    # We combine train and val parquet files to get the complete transaction timeline
    train_meta = pd.read_parquet(config.TRAIN_DATA_PATH)
    val_meta = pd.read_parquet(config.VAL_DATA_PATH)

    df_history = pd.concat([train_meta, val_meta], axis=0, ignore_index=True)

    # Load metadata
    df_articles = pd.read_csv(config.ARTICLES_PATH)
    df_customers = pd.read_csv(config.CUSTOMERS_PATH)

    # Preprocess IDs
    # Ensure article_id is string and zero-padded in articles df (it is int64 in raw csv)
    # The metadata parquet files already have article_id as padded strings.
    df_articles["article_id"] = df_articles["article_id"].astype(str).str.zfill(10)

    # Ensure article_id is string in history (safety check)
    if df_history["article_id"].dtype != "object":
        df_history["article_id"] = df_history["article_id"].astype(str).str.zfill(10)

    # Optimize memory usage
    df_history = utils.reduce_mem_usage(df_history)
    df_articles = utils.reduce_mem_usage(df_articles)
    df_customers = utils.reduce_mem_usage(df_customers)

    print(f"Total history rows: {len(df_history)}")
    print(f"Total articles: {len(df_articles)}")
    print(f"Total customers: {len(df_customers)}")

    return df_history, df_articles, df_customers


def get_time_split(df, val_days=None):
    """
    Splits the transaction data into train and validation sets based on the last `val_days`.

    Args:
        df (pd.DataFrame): The transaction dataframe containing 't_dat'.
        val_days (int, optional): Number of days for validation set. Defaults to config.VAL_DAYS.

    Returns:
        tuple: (train_df, val_df)
    """
    if val_days is None:
        val_days = config.VAL_DAYS

    print(f"Splitting data based on last {val_days} days...")

    # Ensure datetime format
    if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    max_date = df["t_dat"].max()
    split_date = max_date - pd.Timedelta(days=val_days)

    train_df = df[df["t_dat"] <= split_date].copy()
    val_df = df[df["t_dat"] > split_date].copy()

    print(
        f"Train range: {train_df['t_dat'].min()} to {train_df['t_dat'].max()} ({len(train_df)} rows)"
    )
    print(
        f"Val range:   {val_df['t_dat'].min()} to {val_df['t_dat'].max()} ({len(val_df)} rows)"
    )

    return train_df, val_df


@utils.cache_result(file_path=config.WORKING_DIR / "processed_sequences.pt")
def preprocess_sequences(df, min_history=3, max_seq_len=20, load_cached_data=False):
    """
    Converts transaction history into padded sequences for the Transformer model.

    Args:
        df (pd.DataFrame): Transaction dataframe.
        min_history (int): Minimum number of transactions required to include a user.
        max_seq_len (int): Maximum sequence length (truncates older items).
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        dict: Contains 'sequences' (Tensor), 'customer_ids' (np.array),
              'article_map' (dict), 'reverse_article_map' (dict), 'vocab_size' (int).
    """
    print("Preprocessing sequences for Sequential Model...")

    # Ensure datetime
    if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    # Sort by customer and time to ensure sequence order
    df_sorted = df.sort_values(["customer_id", "t_dat"])

    # Filter users with sufficient history
    user_counts = df_sorted["customer_id"].value_counts()
    valid_users = user_counts[user_counts >= min_history].index

    print(
        f"Filtering users with < {min_history} transactions. "
        f"Kept {len(valid_users)}/{len(user_counts)} users."
    )

    df_filtered = df_sorted[df_sorted["customer_id"].isin(valid_users)].copy()

    # Create Article Mapping (Vocabulary)
    # We map all articles present in the filtered history
    unique_articles = df_filtered["article_id"].unique()
    # Reserve 0 for padding, so start index at 1
    article_map = {aid: i + 1 for i, aid in enumerate(unique_articles)}
    reverse_article_map = {i + 1: aid for i, aid in enumerate(unique_articles)}
    vocab_size = len(article_map) + 1

    print(f"Vocabulary size: {vocab_size} (including padding)")

    # Map articles to integer indices
    df_filtered["article_idx"] = df_filtered["article_id"].map(article_map)

    # Group by customer and collect sequences
    print("Grouping transactions by user...")
    grouped = df_filtered.groupby("customer_id")["article_idx"].apply(list)

    sequences = []
    customer_ids = []

    print(f"Padding sequences to length {max_seq_len}...")
    for cust_id, seq in grouped.items():
        # Truncate to keep the most recent 'max_seq_len' items
        seq = seq[-max_seq_len:]

        # Left Padding: [0, 0, ..., item_1, item_2]
        # This aligns the last item to the last position, which is standard for
        # many sequential models (SASRec often uses this or masking).
        pad_len = max_seq_len - len(seq)
        padded_seq = [0] * pad_len + seq

        sequences.append(padded_seq)
        customer_ids.append(cust_id)

    # Convert to PyTorch Tensor
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    customer_ids_arr = np.array(customer_ids)

    return {
        "sequences": sequences_tensor,
        "customer_ids": customer_ids_arr,
        "article_map": article_map,
        "reverse_article_map": reverse_article_map,
        "vocab_size": vocab_size,
    }
