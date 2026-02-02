import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    TRANSACTIONS_CACHE_PATH,
    ITEM_MAP_PATH,
    USER_HISTORY_PATH,
    N_WEEKS,
    SEED,
)

# Set fixed random seeds
np.random.seed(SEED)


class TransactionLoader:
    """
    Loads and preprocesses transaction data with temporal filtering.
    """

    def __init__(self):
        self.n_weeks = N_WEEKS

    def load_transactions(self, load_cached_data=True):
        """
        Loads transactions, filters for the last N_WEEKS, and calculates recency.

        Args:
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            pd.DataFrame: Processed transactions DataFrame.
        """
        if load_cached_data and os.path.exists(TRANSACTIONS_CACHE_PATH):
            print(f"Loading cached transactions from {TRANSACTIONS_CACHE_PATH}...")
            return pd.read_parquet(TRANSACTIONS_CACHE_PATH)

        print("Processing transactions from scratch...")
        # Load raw metadata
        df_train = pd.read_csv(TRAIN_PATH)
        df_val = pd.read_csv(VAL_PATH)

        # Combine train and val to get full history for the window
        # We only need specific columns for history building
        cols = ["t_dat", "customer_id", "article_id"]
        df = pd.concat([df_train[cols], df_val[cols]], axis=0, ignore_index=True)

        # Convert date
        df["t_dat"] = pd.to_datetime(df["t_dat"])

        # Determine cutoff date
        max_date = df["t_dat"].max()
        cutoff_date = max_date - pd.Timedelta(weeks=self.n_weeks)

        # Filter by date
        df = df[df["t_dat"] > cutoff_date].copy()

        # Calculate days elapsed (0 = most recent day in data)
        # Adding 1 to avoid division by zero later if needed, though formula usually handles it
        df["days_elapsed"] = (max_date - df["t_dat"]).dt.days

        # Optimize types
        df["article_id"] = df["article_id"].astype("int32")
        df["days_elapsed"] = df["days_elapsed"].astype("int16")

        # Cache the result
        print(f"Saving processed transactions to {TRANSACTIONS_CACHE_PATH}...")
        df.to_parquet(TRANSACTIONS_CACHE_PATH, index=False)

        return df


class IndexMapper:
    """
    Manages mapping between raw IDs (customer_id, article_id) and matrix indices.
    """

    def __init__(self):
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}

    def fit(self, transactions_df, test_df):
        """
        Creates mappings based on Test Users (rows) and Active Items (columns).

        Args:
            transactions_df (pd.DataFrame): Filtered transactions.
            test_df (pd.DataFrame): Test set containing all customers to predict for.
        """
        print("Fitting IndexMapper...")

        # 1. User Mapping
        # Rows must correspond strictly to customers in the submission file
        unique_users = test_df["customer_id"].unique()
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}

        # 2. Item Mapping
        # Columns correspond to items appearing in the training window
        unique_items = transactions_df["article_id"].unique()
        self.item2idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx2item = {idx: i for i, idx in self.item2idx.items()}

        print(
            f"Mapper defined: {len(self.user2idx)} users, {len(self.item2idx)} items."
        )

        # Save Item Map for other modules (Visual/Behavior matrices need to align)
        item_map_df = pd.DataFrame(
            {
                "article_id": list(self.item2idx.keys()),
                "item_idx": list(self.item2idx.values()),
            }
        )
        item_map_df.to_parquet(ITEM_MAP_PATH, index=False)
        print(f"Item map saved to {ITEM_MAP_PATH}")

    def get_num_users(self):
        return len(self.user2idx)

    def get_num_items(self):
        return len(self.item2idx)


class UserHistoryBuilder:
    """
    Constructs the sparse User-Item history matrix with time decay.
    """

    def build_history(self, transactions_df, mapper, load_cached_data=True):
        """
        Builds the sparse interaction matrix.

        Args:
            transactions_df (pd.DataFrame): Processed transactions.
            mapper (IndexMapper): Fitted mapper instance.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            scipy.sparse.csr_matrix: The user history matrix (Users x Items).
        """
        if load_cached_data and os.path.exists(USER_HISTORY_PATH):
            print(f"Loading user history matrix from {USER_HISTORY_PATH}...")
            # Load COO components from parquet and reconstruct CSR
            df_hist = pd.read_parquet(USER_HISTORY_PATH)

            # Reconstruct sparse matrix
            # Shape is derived from mapper
            n_users = mapper.get_num_users()
            n_items = mapper.get_num_items()

            matrix = sp.csr_matrix(
                (
                    df_hist["weight"].values,
                    (df_hist["user_idx"].values, df_hist["item_idx"].values),
                ),
                shape=(n_users, n_items),
            )
            return matrix

        print("Building user history matrix from scratch...")

        # Filter transactions to only include known items and known users
        # (Users in train but not in test are excluded from the U_hist vector rows)
        # (Items not in the active window are already filtered out of transactions_df,
        # but we check against mapper just in case)

        # We need to map IDs to indices.
        # Using map is faster than apply for large series

        # Filter for users in test set
        valid_users = transactions_df["customer_id"].isin(mapper.user2idx)
        df = transactions_df[valid_users].copy()

        if df.empty:
            print("Warning: No history found for test users in the selected window.")
            n_users = mapper.get_num_users()
            n_items = mapper.get_num_items()
            return sp.csr_matrix((n_users, n_items), dtype=np.float32)

        # Map IDs
        # We use a temporary series mapping for speed
        print("Mapping IDs to indices...")
        df["user_idx"] = df["customer_id"].map(mapper.user2idx)
        df["item_idx"] = df["article_id"].map(mapper.item2idx)

        # Drop rows where item_idx is NaN (items not in mapper's item set)
        df = df.dropna(subset=["item_idx"])

        # Cast to integers
        df["user_idx"] = df["user_idx"].astype("int32")
        df["item_idx"] = df["item_idx"].astype("int32")

        # Calculate Time Decay Weight
        # Formula: 1 / (days_elapsed + 1)
        # We add a small epsilon or 1.0 to avoid division by zero if days_elapsed is 0
        print("Calculating time decay weights...")
        df["weight"] = 1.0 / (df["days_elapsed"] + 1.0)

        # Aggregate duplicates
        # If a user bought the same item multiple times, we sum the weights
        print("Aggregating duplicate interactions...")
        df_agg = df.groupby(["user_idx", "item_idx"])["weight"].sum().reset_index()

        # Save to Parquet as COO format
        print(f"Saving user history to {USER_HISTORY_PATH}...")
        df_agg.to_parquet(USER_HISTORY_PATH, index=False)

        # Build CSR Matrix
        print("Constructing CSR matrix...")
        n_users = mapper.get_num_users()
        n_items = mapper.get_num_items()

        matrix = sp.csr_matrix(
            (
                df_agg["weight"].values,
                (df_agg["user_idx"].values, df_agg["item_idx"].values),
            ),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        return matrix
