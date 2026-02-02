import os
import numpy as np
import pandas as pd
from library.config import Config


class TrendEngine:
    """
    Calculates popularity-based baselines (Strata 3 and 4).
    Handles Global Trends and Cohort-based (Age-based) Trends.
    """

    def __init__(self):
        self.config = Config
        self.cache_dir = self.config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_global_trends(self, train_df, indexer, load_cached_data=True):
        """
        Calculates the global time-decayed popularity vector.

        Args:
            train_df (pd.DataFrame): Transactions with 'article_id' and 'days_elapsed'.
            indexer (Indexer): Object containing item mappings.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            np.ndarray: Dense vector of shape (n_items,) with scaled scores.
        """
        cache_path = os.path.join(self.cache_dir, "global_trends.npy")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Global Trends from {cache_path}...")
            return np.load(cache_path)

        print("Calculating Global Trends from scratch...")

        # Filter to only items in the indexer to ensure shape alignment
        valid_items_mask = train_df["article_id"].isin(indexer.item_to_idx)
        df_filtered = train_df[valid_items_mask].copy()

        # Compute scores: Sum of 1.0 / (days_elapsed + epsilon)
        df_filtered["weight"] = 1.0 / (df_filtered["days_elapsed"] + 1e-5)
        pop_series = df_filtered.groupby("article_id")["weight"].sum()

        # Map to dense vector
        n_items = len(indexer.item_to_idx)
        vec = np.zeros(n_items, dtype=np.float32)

        # Get indices and values
        article_ids = pop_series.index
        scores = pop_series.values

        # Map article_ids to indices
        indices = [indexer.item_to_idx[aid] for aid in article_ids]

        # Fill vector
        vec[indices] = scores

        # Normalize to [0, 1]
        max_val = vec.max()
        if max_val > 0:
            vec = vec / max_val

        # Scale
        vec = vec * self.config.SCALE_GLOBAL

        print(f"Global Trends calculated. Max score: {vec.max():.4f}")

        # Cache
        print(f"Caching Global Trends to {cache_path}...")
        np.save(cache_path, vec)

        return vec

    def get_cohort_trends(self, train_df, customers_df, indexer, load_cached_data=True):
        """
        Calculates time-decayed popularity vectors for each age cohort.

        Args:
            train_df (pd.DataFrame): Transactions.
            customers_df (pd.DataFrame): Customer metadata (for age).
            indexer (Indexer): Object containing item mappings.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            dict: Mapping from bin_index (int) to score vector (np.ndarray).
        """
        cache_path = os.path.join(self.cache_dir, "cohort_trends.npz")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Cohort Trends from {cache_path}...")
            loaded = np.load(cache_path)
            # Convert keys back to int (npz stores keys as strings)
            return {int(k): loaded[k] for k in loaded.files}

        print("Calculating Cohort Trends from scratch...")

        # Prepare Data: Merge age into transactions
        cust_subset = customers_df[["customer_id", "age"]].copy()
        df = train_df.merge(cust_subset, on="customer_id", how="left")

        # Handle missing age (fill with -1)
        df["age"] = df["age"].fillna(-1)

        # Binning: Must match logic in data_utils/config
        bins = [-2, 0, 18, 25, 35, 45, 55, 65, 100]
        # labels=False returns 0, 1, 2...
        df["age_bin"] = pd.cut(df["age"], bins=bins, labels=False).astype(int)

        # Pre-calculate weights
        df["weight"] = 1.0 / (df["days_elapsed"] + 1e-5)

        # Filter to valid items
        df = df[df["article_id"].isin(indexer.item_to_idx)]

        cohort_trends = {}
        unique_bins = sorted(df["age_bin"].unique())
        n_items = len(indexer.item_to_idx)

        print(f"Processing {len(unique_bins)} cohorts...")

        for bin_id in unique_bins:
            # Filter for this cohort
            group = df[df["age_bin"] == bin_id]

            if group.empty:
                continue

            # Aggregate
            pop_series = group.groupby("article_id")["weight"].sum()

            # Create vector
            vec = np.zeros(n_items, dtype=np.float32)

            article_ids = pop_series.index
            scores = pop_series.values

            indices = [indexer.item_to_idx[aid] for aid in article_ids]
            vec[indices] = scores

            # Normalize
            max_val = vec.max()
            if max_val > 0:
                vec = vec / max_val

            # Scale
            vec = vec * self.config.SCALE_COHORT

            cohort_trends[bin_id] = vec

        print(f"Cohort Trends calculated for {len(cohort_trends)} bins.")

        # Cache: Save as npz (keys must be strings for savez)
        save_dict = {str(k): v for k, v in cohort_trends.items()}
        print(f"Caching Cohort Trends to {cache_path}...")
        np.savez(cache_path, **save_dict)

        return cohort_trends
