import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
from pathlib import Path
from tqdm import tqdm
from library import config
from library import utils

# ==========================================
# 1. GLOBAL POPULARITY (TREND)
# ==========================================


def get_global_popularity(df, days=None, top_k=12):
    """
    Calculates the most popular articles in the recent time window.

    Args:
        df (pd.DataFrame): Transaction history.
        days (int): Number of days to look back. Defaults to config.TREND_WINDOW_DAYS.
        top_k (int): Number of top items to return.

    Returns:
        list: Top k article_ids (strings).
    """
    if days is None:
        days = config.TREND_WINDOW_DAYS

    # Ensure datetime
    if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    max_date = df["t_dat"].max()
    cutoff_date = max_date - pd.Timedelta(days=days)

    recent_df = df[df["t_dat"] > cutoff_date]

    # Count sales
    pop_counts = recent_df["article_id"].value_counts()

    top_items = pop_counts.head(top_k).index.tolist()
    return top_items


# ==========================================
# 2. REPURCHASE CANDIDATES (HABIT)
# ==========================================


def get_repurchase_candidates(df, customer_ids=None, limit=20):
    """
    Retrieves items previously purchased by the customer, ranked by recency and frequency.

    Args:
        df (pd.DataFrame): Transaction history.
        customer_ids (array-like, optional): Specific customers to retrieve for.
        limit (int): Max items per customer.

    Returns:
        pd.DataFrame: DataFrame with columns ['customer_id', 'article_id']
    """
    # Filter to requested customers if provided
    if customer_ids is not None:
        cust_set = set(customer_ids)
        df = df[df["customer_id"].isin(cust_set)].copy()
    else:
        df = df.copy()

    # Ensure datetime
    if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
        df["t_dat"] = pd.to_datetime(df["t_dat"])

    # Aggregate per user-item pair
    # We want to sort by:
    # 1. Last purchase date (Recency)
    # 2. Purchase count (Frequency)
    user_item_stats = (
        df.groupby(["customer_id", "article_id"])
        .agg(last_date=("t_dat", "max"), count=("article_id", "count"))
        .reset_index()
    )

    # Sort
    user_item_stats = user_item_stats.sort_values(
        ["customer_id", "last_date", "count"], ascending=[True, False, False]
    )

    # Take top K per user
    repurchase_df = user_item_stats.groupby("customer_id").head(limit)

    return repurchase_df[["customer_id", "article_id"]]


# ==========================================
# 3. CO-OCCURRENCE MATRIX (STRUCTURE)
# ==========================================


class CooccurrenceMatrix:
    def __init__(self):
        self.matrix = None
        self.article_map = None
        self.reverse_map = None
        self.is_fitted = False

    def fit(self, df, weeks=None, load_cached_data=True):
        """
        Builds or loads the co-occurrence matrix.
        """
        if weeks is None:
            weeks = config.COOC_HISTORY_WEEKS

        # Define cache paths
        matrix_path = config.COOC_MATRIX_PATH
        map_path = config.WORKING_DIR / "cooc_article_map.npy"

        # Try loading cache
        if load_cached_data and matrix_path.exists() and map_path.exists():
            print(f"Loading cached Cooccurrence Matrix from {matrix_path}...")
            try:
                self.matrix = sp.load_npz(matrix_path)
                self.article_map = np.load(map_path, allow_pickle=True).item()
                self.reverse_map = {v: k for k, v in self.article_map.items()}
                self.is_fitted = True
                return
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        # Compute from scratch
        print("Building Cooccurrence Matrix from scratch...")

        # 1. Filter Data
        if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
            df["t_dat"] = pd.to_datetime(df["t_dat"])

        max_date = df["t_dat"].max()
        cutoff_date = max_date - pd.Timedelta(weeks=weeks)

        # We only care about transactions in the window
        df_filtered = df[df["t_dat"] > cutoff_date].copy()

        # 2. Map IDs
        # We need integer indices for sparse matrix
        unique_users = df_filtered["customer_id"].unique()
        unique_items = df_filtered["article_id"].unique()

        user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.article_map = {aid: i for i, aid in enumerate(unique_items)}
        self.reverse_map = {i: aid for i, aid in enumerate(unique_items)}

        # Apply mapping
        user_indices = df_filtered["customer_id"].map(user_map)
        item_indices = df_filtered["article_id"].map(self.article_map)

        # 3. Calculate Weights (Inverse Time Decay)
        # w = 1 / (days_elapsed + 1)
        # days_elapsed = (max_date - t_dat).days
        df_filtered["days_elapsed"] = (max_date - df_filtered["t_dat"]).dt.days
        weights = 1.0 / (df_filtered["days_elapsed"] + 1.0)

        # 4. Create User-Item Matrix (M)
        # Shape: (n_users, n_items)
        # We sum weights if user bought same item multiple times in window
        M = sp.coo_matrix(
            (weights, (user_indices, item_indices)),
            shape=(len(unique_users), len(unique_items)),
        ).tocsr()

        # 5. Compute Co-occurrence (C = M^T * M)
        # Shape: (n_items, n_items)
        # This gives the weighted similarity between items based on shared user history
        print("Computing M^T * M...")
        self.matrix = M.T.dot(M)

        # Zero out diagonal (item co-occurring with itself is trivial)
        self.matrix.setdiag(0)
        self.matrix.eliminate_zeros()

        # 6. Save Cache
        print("Saving Cooccurrence Matrix to cache...")
        sp.save_npz(matrix_path, self.matrix)
        np.save(map_path, self.article_map)

        self.is_fitted = True

    def get_candidates(self, df_history, customer_ids, top_k=12, batch_size=1000):
        """
        Generates candidates for specific customers based on their history.

        Args:
            df_history (pd.DataFrame): Full transaction history.
            customer_ids (list): List of customer_ids to predict for.
            top_k (int): Number of candidates to retrieve.
            batch_size (int): Number of customers to process at once.

        Returns:
            dict: {customer_id: [article_id_1, article_id_2, ...]}
        """
        if not self.is_fitted:
            raise ValueError("Matrix not fitted. Call fit() first.")

        print(
            f"Generating Cooccurrence candidates for {len(customer_ids)} customers..."
        )

        # Filter history to relevant customers and items known to the matrix
        # We can't predict based on items we haven't seen in the cooc window
        relevant_history = df_history[
            (df_history["customer_id"].isin(customer_ids))
            & (df_history["article_id"].isin(self.article_map))
        ].copy()

        # Map IDs
        # Note: We create a local user map just for this batch of targets
        target_cust_map = {cid: i for i, cid in enumerate(customer_ids)}

        relevant_history["user_idx"] = relevant_history["customer_id"].map(
            target_cust_map
        )
        relevant_history["item_idx"] = relevant_history["article_id"].map(
            self.article_map
        )

        # Drop any rows where mapping failed (though isin check should handle it)
        relevant_history = relevant_history.dropna(subset=["user_idx", "item_idx"])

        # Create User-Item History Matrix for targets
        # Binary interaction is usually sufficient for query vector,
        # but we can use frequency/recency if desired. Let's use binary for query.
        # Shape: (n_targets, n_known_items)
        U_target = sp.csr_matrix(
            (
                np.ones(len(relevant_history)),
                (relevant_history["user_idx"], relevant_history["item_idx"]),
            ),
            shape=(len(customer_ids), len(self.article_map)),
        )

        results = {}

        # Process in batches to avoid OOM on dense result
        for start_idx in range(0, len(customer_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(customer_ids))

            # Slice batch of users
            U_batch = U_target[start_idx:end_idx]

            # Matrix Multiply: (Batch, Items) * (Items, Items) -> (Batch, Items)
            # Result contains the summed co-occurrence scores for each item
            scores = U_batch.dot(self.matrix)

            # Extract top K
            # We iterate through the sparse/dense result rows
            for i in range(scores.shape[0]):
                row = scores[i]

                # If row is sparse, convert to dense for argsort, or use sparse logic
                # Given top_k is small, dense is fine for 100k items (vector size ~400KB)
                if sp.issparse(row):
                    row = row.toarray().flatten()

                # Get indices of top scores
                # argsort is ascending, so take last k and reverse
                if row.sum() == 0:
                    top_indices = []
                else:
                    # optimization: argpartition is faster than argsort for top k
                    if len(row) > top_k:
                        top_indices = np.argpartition(row, -top_k)[-top_k:]
                        # Sort these top k strictly
                        top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
                    else:
                        top_indices = np.argsort(row)[::-1]

                # Map back to article IDs
                candidates = [self.reverse_map.get(idx) for idx in top_indices]

                # Store
                real_cust_idx = start_idx + i
                cust_id = customer_ids[real_cust_idx]
                results[cust_id] = candidates

        return results
