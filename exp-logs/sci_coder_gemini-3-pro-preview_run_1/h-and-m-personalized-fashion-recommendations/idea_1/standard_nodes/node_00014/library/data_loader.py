import pandas as pd
import numpy as np
import os
from datetime import timedelta
from library.config import Config
from library.utils import reduce_mem_usage


class IdMapper:
    """
    Handles mapping between original IDs (customer_id strings, article_id ints)
    and dense integer indices for efficient processing.
    """

    def __init__(self):
        self.customer_to_idx = {}
        self.idx_to_customer = {}
        self.article_to_idx = {}
        self.idx_to_article = {}
        self.is_fitted = False

    def fit(self, customers_df, articles_df):
        """
        Creates mappings based on unique customers and articles.
        """
        # Customers
        unique_customers = customers_df["customer_id"].unique()
        self.customer_to_idx = {cid: i for i, cid in enumerate(unique_customers)}
        self.idx_to_customer = {i: cid for i, cid in enumerate(unique_customers)}

        # Articles
        unique_articles = articles_df["article_id"].unique()
        self.article_to_idx = {aid: i for i, aid in enumerate(unique_articles)}
        self.idx_to_article = {i: aid for i, aid in enumerate(unique_articles)}

        self.is_fitted = True

    def transform_customers(self, df, col="customer_id"):
        """Maps customer_id column to dense integers."""
        return df[col].map(self.customer_to_idx).fillna(-1).astype("int32")

    def transform_articles(self, df, col="article_id"):
        """Maps article_id column to dense integers."""
        return df[col].map(self.article_to_idx).fillna(-1).astype("int32")

    def inverse_transform_articles(self, indices):
        """Maps dense integer indices back to original article_ids."""
        # Vectorized lookup using numpy array if possible, or list comprehension
        # Since idx_to_article is a dict, list comp is safe
        return [self.idx_to_article.get(i, -1) for i in indices]

    def save(self, cache_dir):
        """Saves mappings to parquet files."""
        os.makedirs(cache_dir, exist_ok=True)

        # Save Customer Map
        cust_df = pd.DataFrame(
            {
                "customer_id": list(self.customer_to_idx.keys()),
                "idx": list(self.customer_to_idx.values()),
            }
        )
        cust_df.to_parquet(
            os.path.join(cache_dir, "map_customers.parquet"), index=False
        )

        # Save Article Map
        art_df = pd.DataFrame(
            {
                "article_id": list(self.article_to_idx.keys()),
                "idx": list(self.article_to_idx.values()),
            }
        )
        art_df.to_parquet(os.path.join(cache_dir, "map_articles.parquet"), index=False)

    def load(self, cache_dir):
        """Loads mappings from parquet files."""
        cust_path = os.path.join(cache_dir, "map_customers.parquet")
        art_path = os.path.join(cache_dir, "map_articles.parquet")

        if not (os.path.exists(cust_path) and os.path.exists(art_path)):
            raise FileNotFoundError("Cached mapping files not found.")

        cust_df = pd.read_parquet(cust_path)
        self.customer_to_idx = dict(zip(cust_df["customer_id"], cust_df["idx"]))
        self.idx_to_customer = dict(zip(cust_df["idx"], cust_df["customer_id"]))

        art_df = pd.read_parquet(art_path)
        self.article_to_idx = dict(zip(art_df["article_id"], art_df["idx"]))
        self.idx_to_article = dict(zip(art_df["idx"], art_df["article_id"]))

        self.is_fitted = True


def load_filtered_transactions(
    train_files=None,
    customers_path=Config.CUSTOMERS_PATH,
    articles_path=Config.ARTICLES_PATH,
    weeks=Config.HISTORY_WEEKS,
    load_cached_data=True,
):
    """
    Loads transactions, filters them by time, and maps IDs to integers.
    Uses caching to speed up subsequent runs.

    Args:
        train_files (list): List of paths to transaction CSVs. Defaults to [Config.TRAIN_DATA_PATH].
        customers_path (str): Path to customers metadata.
        articles_path (str): Path to articles metadata.
        weeks (int): Number of weeks of history to keep.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (pd.DataFrame, IdMapper)
            - DataFrame with columns: ['t_dat', 'customer_id', 'article_id', 'days_elapsed']
              where IDs are dense integers.
            - Fitted IdMapper instance.
    """
    if train_files is None:
        train_files = [Config.TRAIN_DATA_PATH]

    # Define cache paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Include weeks in filename to invalidate cache if window changes
    cache_file_name = f"transactions_w{weeks}.parquet"
    cache_path = os.path.join(cache_dir, cache_file_name)

    mapper = IdMapper()

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            df = pd.read_parquet(cache_path)

            print("Loading cached ID mappings...")
            mapper.load(cache_dir)

            return df, mapper
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print("Processing data from scratch...")

    # A. Initialize and Fit Mapper
    print("Loading metadata for ID mapping...")
    # Load raw customers and articles to ensure global mapping coverage
    # (needed for test set prediction even if users not in train)
    cust_df = pd.read_csv(customers_path, usecols=["customer_id"])
    art_df = pd.read_csv(
        articles_path, usecols=["article_id"], dtype={"article_id": "int32"}
    )

    print("Fitting ID Mapper...")
    mapper.fit(cust_df, art_df)

    # B. Load and Concatenate Transactions
    print(f"Loading transactions from {train_files}...")
    dfs = []
    for path in train_files:
        # Load only necessary columns
        d = pd.read_csv(
            path,
            usecols=["t_dat", "customer_id", "article_id"],
            dtype={"article_id": "int32"},
        )
        dfs.append(d)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    # C. Date Processing & Filtering
    print("Converting dates...")
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    max_date = df["t_dat"].max()
    cutoff_date = max_date - timedelta(weeks=weeks)

    print(
        f"Filtering transactions (Max Date: {max_date.date()}, Cutoff: {cutoff_date.date()})..."
    )
    df = df[df["t_dat"] >= cutoff_date].copy()

    # Calculate days elapsed (0 = most recent day in data)
    # We use (max_date - t_dat).days.
    # If t_dat == max_date, days_elapsed = 0.
    df["days_elapsed"] = (max_date - df["t_dat"]).dt.days.astype("int16")

    # D. Map IDs
    print("Mapping IDs to dense integers...")
    df["customer_id"] = mapper.transform_customers(df)
    df["article_id"] = mapper.transform_articles(df)

    # Remove rows where mapping failed (should be 0 if fit on full metadata)
    original_len = len(df)
    df = df[(df["customer_id"] != -1) & (df["article_id"] != -1)]
    if len(df) < original_len:
        print(f"Dropped {original_len - len(df)} rows due to unmapped IDs.")

    # E. Optimization
    df = reduce_mem_usage(df)

    # F. Save to Cache
    print("Saving data and mappings to cache...")
    df.to_parquet(cache_path, index=False)
    mapper.save(cache_dir)

    return df, mapper
