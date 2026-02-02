import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import gc
from library.config import Config
from library.sparse_engine import SparseEngine


class DWSCRecommender:
    """
    Decay-Weighted Stratified Cascade (DWSC) Recommender.

    Implements a three-stratum retrieval system:
    1. Habit (Repurchase): High priority, strict time decay.
    2. CF (Discovery): Medium priority, item-item similarity with decay.
    3. Trend (Fallback): Low priority, global popularity.
    """

    def __init__(self):
        self.sparse_engine = SparseEngine()
        self.S = None  # Similarity Matrix (Items x Items)
        self.X_cf = None  # Interaction Matrix for CF (Users x Items)
        self.trend_scores = None  # Global Trend Vector (Items)
        self.n_users = 0
        self.n_items = 0
        self.dtype = np.float32 if Config.PRECISION == "float32" else np.float64

    def fit(self, train_df, n_users, n_items, load_cached_data=True):
        """
        Builds the necessary matrices: X_cf, S, and Global Trend.

        Args:
            train_df: DataFrame containing training transactions.
            n_users: Total number of users in the encoder.
            n_items: Total number of items in the encoder.
            load_cached_data: Whether to load intermediate matrices from disk.
        """
        print("Fitting DWSC Recommender...")
        self.n_users = n_users
        self.n_items = n_items

        # 1. Build CF Interaction Matrix (X_cf)
        # This includes IDF weighting and L2 Normalization (managed by SparseEngine)
        # Used for Stratum 2 (CF)
        self.X_cf = self.sparse_engine.build_decay_matrix(
            train_df, n_users, n_items, load_cached_data=load_cached_data
        )

        # 2. Compute Item-Item Similarity (S)
        # S = X_cf.T @ X_cf (managed by SparseEngine)
        self.S = self.sparse_engine.compute_similarity(
            self.X_cf, load_cached_data=load_cached_data
        )

        # 3. Compute Global Trend (Stratum 3)
        self._compute_global_trend(train_df, n_items, load_cached_data)

        print("Fit complete.")

    def _compute_global_trend(self, train_df, n_items, load_cached_data):
        """
        Computes the global trend vector based on decay-weighted popularity.
        Scores are normalized to [0, 1].
        """
        cache_path = os.path.join(Config.CACHE_DIR, "global_trend.npy")

        if load_cached_data and os.path.exists(cache_path):
            print("Loading cached global trend...")
            self.trend_scores = np.load(cache_path)
            return

        print("Computing global trend vector...")
        # Trend = Sum(1 / (days + 1)) per item
        # No IDF, just raw velocity to capture what is currently moving
        days = train_df["days_elapsed"].values.astype(self.dtype)
        # Use simple 1/t decay for trend
        weights = 1.0 / (days + 1.0)
        item_indices = train_df["item_id"].values

        # Sum weights per item
        trend = np.bincount(item_indices, weights=weights, minlength=n_items)

        # Normalize to [0, 1]
        if trend.max() > 0:
            trend = trend / trend.max()

        self.trend_scores = trend.astype(self.dtype)

        print(f"Caching global trend to {cache_path}...")
        np.save(cache_path, self.trend_scores)

    def _build_habit_matrix(self, train_df, n_users, n_items, load_cached_data=True):
        """
        Builds the Habit Matrix (X_habit) for Stratum 1.
        Logic: Weight = 1 / (days + 1)^power.
        Difference from CF: No IDF, No Normalization, potentially different decay power.
        """
        cache_path = os.path.join(Config.CACHE_DIR, "X_habit.npz")

        if load_cached_data and os.path.exists(cache_path):
            print("Loading cached habit matrix...")
            return sp.load_npz(cache_path)

        print("Building habit matrix...")
        days = train_df["days_elapsed"].values.astype(self.dtype)
        weights = 1.0 / np.power(days + 1.0, Config.HISTORY_DECAY_POWER)

        user_indices = train_df["user_id"].values
        item_indices = train_df["item_id"].values

        # Construct COO then CSR
        # Summing duplicates is desirable here (buying multiple times reinforces habit)
        X_habit = sp.coo_matrix(
            (weights, (user_indices, item_indices)),
            shape=(n_users, n_items),
            dtype=self.dtype,
        ).tocsr()

        print(f"Caching habit matrix to {cache_path}...")
        sp.save_npz(cache_path, X_habit)
        return X_habit

    def predict(self, test_user_ids, train_df, encoder, load_cached_data=True):
        """
        Generates predictions for test_user_ids using the stratified cascade.

        Args:
            test_user_ids: Array of integer user indices to predict for.
            train_df: Training data (needed to build habit matrix).
            encoder: DataEncoder instance for inverse transformation.
            load_cached_data: Whether to use cached habit matrix.

        Returns:
            pd.DataFrame: Submission dataframe.
        """
        print("Starting inference...")

        # 1. Prepare Habit Matrix (Stratum 1 Source)
        X_habit = self._build_habit_matrix(
            train_df, self.n_users, self.n_items, load_cached_data
        )

        # 2. Prepare Output
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        results = []

        # 3. Batch Processing
        batch_size = Config.BATCH_SIZE
        n_test = len(test_user_ids)

        # Pre-scale trend to target range [0, TREND_OFFSET_MAX]
        # Shape: (N_items,)
        scaled_trend = self.trend_scores * Config.TREND_OFFSET_MAX

        print(f"Predicting for {n_test} users in batches of {batch_size}...")

        for start_idx in range(0, n_test, batch_size):
            end_idx = min(start_idx + batch_size, n_test)
            batch_users = test_user_ids[start_idx:end_idx]

            # --- Stratum 2: CF Scores (Discovery) ---
            # Retrieve query vectors from X_cf (built in fit)
            # X_cf contains history for all users (train + test) if they exist in data
            batch_queries = self.X_cf[batch_users]

            # Compute raw similarity scores: (Batch, Items) = (Batch, Items) @ (Items, Items)
            # S is pruned sparse, so this is efficient
            cf_scores = batch_queries.dot(self.S)

            if sp.issparse(cf_scores):
                cf_scores = cf_scores.toarray()

            # Scale CF Scores to [CF_OFFSET_MIN, CF_OFFSET_MAX]
            # We assume normalized dot products are approx [0, 1].
            # We scale linearly.
            cf_scores = (
                cf_scores * (Config.CF_OFFSET_MAX - Config.CF_OFFSET_MIN)
                + Config.CF_OFFSET_MIN
            )

            # --- Stratum 1: Habit Scores (Priors) ---
            # Retrieve history vectors
            habit_scores = X_habit[batch_users].toarray()

            # Apply Stratification: Shift positive history to [HISTORY_OFFSET, inf)
            # This ensures any repurchased item (even from 10 weeks ago) outranks pure CF
            mask = habit_scores > 0
            habit_scores[mask] += Config.HISTORY_OFFSET

            # --- Stratum 3: Trend Scores (Fallback) ---
            # scaled_trend is broadcasted to (Batch, N_items)

            # --- Aggregation ---
            # Total = Habit + CF + Trend
            # Due to disjoint ranges, this enforces: Habit > CF > Trend
            total_scores = habit_scores + cf_scores + scaled_trend

            # --- Retrieval ---
            # Get top K indices efficiently
            k = Config.TOP_K

            # argpartition finds the k largest elements (unsorted)
            # We negate total_scores to use argpartition for largest elements if needed,
            # but standard argpartition with -k gives indices of k largest at the end.
            top_k_indices = np.argpartition(total_scores, -k, axis=1)[:, -k:]

            # Extract scores for these top k to sort them correctly
            rows = np.arange(len(batch_users))[:, None]
            top_k_scores = total_scores[rows, top_k_indices]

            # Sort indices based on scores (descending)
            sort_order = np.argsort(top_k_scores, axis=1)[:, ::-1]
            sorted_indices = top_k_indices[rows, sort_order]

            # --- Formatting ---
            # Convert indices to article IDs
            batch_user_ids = encoder.inverse_transform_users(batch_users)

            for i, u_id in enumerate(batch_user_ids):
                item_indices = sorted_indices[i]
                item_ids = encoder.inverse_transform_items(item_indices)
                pred_str = " ".join(str(x) for x in item_ids)
                results.append((u_id, pred_str))

            # Explicit garbage collection for large batch arrays
            del cf_scores, habit_scores, total_scores, batch_queries

        # Create DataFrame
        submission_df = pd.DataFrame(results, columns=["customer_id", "prediction"])

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

        return submission_df
