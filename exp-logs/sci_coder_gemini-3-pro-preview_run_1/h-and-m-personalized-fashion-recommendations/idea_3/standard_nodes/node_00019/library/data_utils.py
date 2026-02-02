import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from library.config import Config


class IndexMapper:
    """
    Handles mapping between raw IDs (strings/ints) and matrix indices (0..N-1).
    Ensures consistent mapping across Train/Val/Test sets by establishing
    a global universe of users and items.
    """

    def __init__(self, users, items):
        # Sort to ensure deterministic mapping
        self.users = np.array(sorted(list(set(users))))
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}

        self.items = np.array(sorted(list(set(items))))
        self.item_to_idx = {i: idx for idx, i in enumerate(self.items)}

    def get_num_users(self):
        return len(self.users)

    def get_num_items(self):
        return len(self.items)

    def map_users(self, user_series):
        """Maps a series of user IDs to indices. Returns -1 for unknown users."""
        return user_series.map(self.user_to_idx).fillna(-1).astype(np.int32)

    def map_items(self, item_series):
        """Maps a series of item IDs to indices. Returns -1 for unknown items."""
        return item_series.map(self.item_to_idx).fillna(-1).astype(np.int32)

    def get_users_from_indices(self, indices):
        return self.users[indices]

    def get_items_from_indices(self, indices):
        return self.items[indices]


def get_global_mapper():
    """
    Factory function to create a consistent mapper from all available
    customers and articles in the dataset.
    """
    print("Building Global IndexMapper...")
    # Load all customers (Train + Test universe)
    cust_df = pd.read_csv(Config.PATH_CUSTOMERS, usecols=["customer_id"])
    all_customers = cust_df["customer_id"].unique()

    # Load all articles
    art_df = pd.read_csv(
        Config.PATH_ARTICLES, usecols=["article_id"], dtype={"article_id": "int32"}
    )
    all_articles = art_df["article_id"].unique()

    mapper = IndexMapper(all_customers, all_articles)
    print(
        f"Mapper created: {mapper.get_num_users()} users, {mapper.get_num_items()} items."
    )
    return mapper


def load_and_filter_data(
    data_path=Config.PATH_TRAIN, weeks=Config.TRAIN_WEEKS, debug=Config.DEBUG
):
    """
    Loads transaction data and filters for the specified number of recent weeks.

    Args:
        data_path (str): Path to the transaction CSV.
        weeks (int): Number of weeks of history to keep.
        debug (bool): If True, samples a subset for rapid iteration.

    Returns:
        pd.DataFrame: Filtered transactions.
    """
    print(f"Loading data from {data_path}...")

    # Load data with optimized types to save memory
    df = pd.read_csv(
        data_path,
        dtype={"article_id": "int32", "price": "float32", "sales_channel_id": "int8"},
    )

    # Convert date column
    df["t_dat"] = pd.to_datetime(df["t_dat"])

    # Determine cutoff date based on the latest date in the dataset
    max_date = df["t_dat"].max()
    cutoff_date = max_date - pd.Timedelta(weeks=weeks)

    print(f"Filtering data from {cutoff_date} to {max_date} ({weeks} weeks)...")
    df_filtered = df[df["t_dat"] > cutoff_date].copy()

    if debug:
        print(f"DEBUG mode: Sampling {Config.DEBUG_SAMPLES} rows...")
        if len(df_filtered) > Config.DEBUG_SAMPLES:
            df_filtered = df_filtered.sample(
                Config.DEBUG_SAMPLES, random_state=Config.SEED
            )

    print(f"Data loaded. Shape: {df_filtered.shape}")
    return df_filtered


def build_user_history_vectors(df, mapper, load_cached_data=True):
    """
    Constructs sparse User-Item interaction vectors with time-decay weights.

    Weight calculation: 1.0 / (days_elapsed + 1.0)

    Args:
        df (pd.DataFrame): Filtered transaction DataFrame. Required if cache is missing.
        mapper (IndexMapper): Fitted IndexMapper instance.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix (Users x Items) with decay weights.
    """
    cache_path = Config.CACHE_USER_HISTORY

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading user history from cache: {cache_path}")
        try:
            hist_df = pd.read_parquet(cache_path)

            # Reconstruct Sparse Matrix
            n_users = mapper.get_num_users()
            n_items = mapper.get_num_items()

            data = hist_df["weight"].values.astype(np.float32)
            row = hist_df["user_idx"].values
            col = hist_df["item_idx"].values

            matrix = sp.csr_matrix((data, (row, col)), shape=(n_users, n_items))
            print(f"Loaded User History Matrix Shape: {matrix.shape}")
            return matrix
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute if not cached or load failed
    if df is None:
        raise ValueError(
            "DataFrame 'df' is required when cache is missing or load_cached_data=False."
        )

    print("Computing user history vectors...")

    # Calculate Time Decay Weights
    # We assume 't_dat' is already datetime from load_and_filter_data
    max_date = df["t_dat"].max()
    days_diff = (max_date - df["t_dat"]).dt.days

    # Weight formula: Higher weight for more recent purchases
    weights = 1.0 / (days_diff + 1.0)

    # Map IDs to Indices
    print("Mapping IDs to indices...")
    user_indices = mapper.map_users(df["customer_id"])
    item_indices = mapper.map_items(df["article_id"])

    # Create intermediate DataFrame
    hist_df = pd.DataFrame(
        {
            "user_idx": user_indices,
            "item_idx": item_indices,
            "weight": weights.astype(np.float32),
        }
    )

    # Filter out transactions for users/items not in the mapper (if any)
    valid_mask = (hist_df["user_idx"] >= 0) & (hist_df["item_idx"] >= 0)
    if not valid_mask.all():
        print(
            f"Dropping {len(hist_df) - valid_mask.sum()} transactions with unknown IDs."
        )
        hist_df = hist_df[valid_mask]

    # Aggregate duplicates: Sum weights for multiple purchases of the same item by the same user
    # This captures the intensity of user interest
    print("Aggregating duplicates...")
    hist_df = hist_df.groupby(["user_idx", "item_idx"], as_index=False)["weight"].sum()

    # Save to cache
    print(f"Saving user history to cache: {cache_path}")
    hist_df.to_parquet(cache_path, index=False)

    # Convert to Sparse Matrix
    n_users = mapper.get_num_users()
    n_items = mapper.get_num_items()

    data = hist_df["weight"].values
    row = hist_df["user_idx"].values
    col = hist_df["item_idx"].values

    matrix = sp.csr_matrix((data, (row, col)), shape=(n_users, n_items))

    print(f"Computed User History Matrix Shape: {matrix.shape}")
    return matrix
