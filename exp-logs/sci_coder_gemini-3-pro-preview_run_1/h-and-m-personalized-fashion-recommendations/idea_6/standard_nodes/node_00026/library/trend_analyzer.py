import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from library.config import Config


class TrendAnalyzer:
    """
    Analyzes transaction data to generate time-decayed popularity trends.
    Implements Stratum 3 (Cohort Trends) and Stratum 4 (Global Trends) of the SDCC model.
    """

    @staticmethod
    def _calculate_decay_weights(df):
        """
        Calculates time-decay weights for a transaction DataFrame.
        Weight = 1 / (days_elapsed + 1) ** ALPHA
        """
        # Ensure datetime
        if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
            t_dat = pd.to_datetime(df["t_dat"])
        else:
            t_dat = df["t_dat"]

        max_date = t_dat.max()
        days_elapsed = (max_date - t_dat).dt.days.values

        weights = 1.0 / np.power(days_elapsed + 1.0, Config.TIME_DECAY_ALPHA)
        return weights.astype(np.float32)

    @staticmethod
    def compute_global_trends(df, item_to_idx, load_cached_data=True):
        """
        Computes a single time-decayed popularity vector for the entire dataset.
        Scaled to range [0, 9] (approx) for Stratum 4.

        Args:
            df (pd.DataFrame): Filtered transaction DataFrame.
            item_to_idx (dict): Mapping from article_id to item index.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            np.ndarray: Dense array of shape (n_items,) containing global scores.
        """
        cache_path = os.path.join(Config.CACHE_DIR, "global_trends.npy")

        if load_cached_data and os.path.exists(cache_path):
            print("Loading global trends from cache...")
            return np.load(cache_path)

        print("Computing global trends...")

        # Calculate weights
        weights = TrendAnalyzer._calculate_decay_weights(df)

        # Map items
        item_indices = df["article_id"].map(item_to_idx).fillna(-1).astype(np.int32)

        # Filter valid
        mask = item_indices != -1
        valid_indices = item_indices[mask]
        valid_weights = weights[mask]

        # Aggregate
        n_items = len(item_to_idx)
        global_trends = np.zeros(n_items, dtype=np.float32)

        # Use np.add.at for unbuffered summation
        np.add.at(global_trends, valid_indices, valid_weights)

        # Scale to [0, 9]
        # Logic: Normalize max to 1.0, then multiply by 9.0
        max_val = global_trends.max()
        if max_val > 0:
            global_trends = (global_trends / max_val) * 9.0

        print(f"Global trends computed. Shape: {global_trends.shape}")
        print(f"Saving to {cache_path}...")
        np.save(cache_path, global_trends)

        return global_trends

    @staticmethod
    def compute_cohort_trends(
        df, cohort_array, user_to_idx, item_to_idx, load_cached_data=True
    ):
        """
        Computes popularity vectors for each age cohort.
        Scaled to range [0, 80] (approx) for Stratum 3 (which becomes [10, 90] with offset).

        Args:
            df (pd.DataFrame): Filtered transaction DataFrame.
            cohort_array (np.ndarray): Array mapping user_idx -> cohort_idx.
            user_to_idx (dict): Mapping from customer_id to user index.
            item_to_idx (dict): Mapping from article_id to item index.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            dict: Mapping {cohort_id: sp.csr_matrix(1, n_items)}
        """
        cache_path = os.path.join(Config.CACHE_DIR, "cohort_trends.npy")

        if load_cached_data and os.path.exists(cache_path):
            print("Loading cohort trends from cache...")
            return np.load(cache_path, allow_pickle=True).item()

        print("Computing cohort trends...")

        # Calculate weights
        weights = TrendAnalyzer._calculate_decay_weights(df)

        # Map IDs
        user_indices = df["customer_id"].map(user_to_idx).fillna(-1).astype(np.int32)
        item_indices = df["article_id"].map(item_to_idx).fillna(-1).astype(np.int32)

        # Filter valid
        mask = (user_indices != -1) & (item_indices != -1)
        valid_user_idx = user_indices[mask]
        valid_item_idx = item_indices[mask]
        valid_weights = weights[mask]

        # Get cohorts for transactions
        # cohort_array is aligned with user_idx
        valid_cohorts = cohort_array[valid_user_idx]

        # Aggregate
        # We construct a matrix (n_cohorts, n_items)
        n_items = len(item_to_idx)
        n_cohorts = cohort_array.max() + 1

        # Use sparse matrix construction for aggregation
        # Rows: cohort_idx, Cols: item_idx
        cohort_matrix = sp.csr_matrix(
            (valid_weights, (valid_cohorts, valid_item_idx)),
            shape=(n_cohorts, n_items),
            dtype=np.float32,
        )

        # Convert to dictionary of normalized sparse vectors
        cohort_trends = {}

        print(f"Processing {n_cohorts} cohorts...")
        for c_idx in range(n_cohorts):
            # Extract row
            row_vec = cohort_matrix.getrow(c_idx)

            # Scale to [0, 80]
            if row_vec.nnz > 0:
                max_val = row_vec.data.max()
                if max_val > 0:
                    row_vec.data = (row_vec.data / max_val) * 80.0

            cohort_trends[c_idx] = row_vec

        print(f"Cohort trends computed. Saving to {cache_path}...")
        np.save(cache_path, cohort_trends)

        return cohort_trends
