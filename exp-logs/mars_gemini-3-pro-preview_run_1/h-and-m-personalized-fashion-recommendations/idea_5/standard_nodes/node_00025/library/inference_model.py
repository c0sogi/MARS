import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import gc
from sklearn.preprocessing import normalize
from library import config
from library import data_processor
from library import graph_engine


class StratifiedRecommender:
    """
    Implements the Stratified Vectorized Hybrid-Graph Cascade recommendation system.
    Generates predictions by layering three distinct signal strata:
    1. Habitual Repurchase (History) -> Score range [1000, inf)
    2. Hybrid Collaborative Filtering -> Score range [10, 900]
    3. Global Trends (Popularity)    -> Score range [0, 9]
    """

    def __init__(self):
        self.working_dir = config.WORKING_DIR
        self.submission_path = config.SUBMISSION_PATH
        self.precision = config.PRECISION

        # Data and Mappings
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None

        # Model Components
        self.S_hybrid = None  # Stratum 2 Matrix
        self.global_trends = None  # Stratum 3 Vector
        self.user_history = None  # Stratum 1 Matrix (and Query for Stratum 2)

    def load_resources(self, load_cached_data=True):
        """
        Loads all necessary data, mappings, and model artifacts.
        """
        print("Initializing StratifiedRecommender resources...")

        # 1. Load Base Data & Mappings
        loader = data_processor.DataLoader()
        train_df, val_df, test_df, articles_df, user_map, item_map = loader.load_data(
            load_cached_data=load_cached_data
        )

        self.user_map = user_map
        self.item_map = item_map
        # Create reverse map for decoding predictions (int -> str)
        self.reverse_item_map = pd.Series(
            item_map.index.values, index=item_map.values
        ).to_dict()

        # 2. Load/Compute Hybrid Similarity Matrix (Stratum 2)
        optimizer = graph_engine.SimilarityOptimizer()
        self.S_hybrid = optimizer.run(
            train_df, articles_df, user_map, item_map, load_cached_data=load_cached_data
        )

        # 3. Load/Compute Global Trends (Stratum 3)
        self.global_trends = self._get_global_trends(train_df, load_cached_data)

        # 4. Load/Compute User History Vectors (Stratum 1)
        self.user_history = self._get_user_history(
            train_df, len(user_map), len(item_map), load_cached_data
        )

        return train_df, val_df, test_df

    def _get_global_trends(self, train_df, load_cached_data):
        """
        Computes or loads the Global Trend vector.
        Score = Sum(1 / (days_elapsed + 1)^alpha)
        Scaled to [0, 9].
        """
        cache_path = config.CACHE_GLOBAL_TRENDS

        if load_cached_data and os.path.exists(cache_path):
            print("Loading global trends from cache...")
            try:
                df = pd.read_parquet(cache_path)
                trends = np.zeros(len(self.item_map), dtype=self.precision)
                # Map scores back to the correct index
                trends[df["item_idx"].values] = df["score"].values
                return trends
            except Exception as e:
                print(f"Error loading trends cache: {e}. Recomputing...")

        print("Computing global trends...")
        # Ensure days_elapsed is calculated
        if "days_elapsed" not in train_df.columns:
            max_date = train_df["t_dat"].max()
            train_df["days_elapsed"] = (max_date - train_df["t_dat"]).dt.days

        # Compute decay score
        alpha = config.TREND_DECAY_ALPHA
        # Work on a slice to avoid SettingWithCopy warnings if train_df is a view
        temp_df = train_df[["item_idx", "days_elapsed"]].copy()
        temp_df["score"] = 1.0 / np.power(temp_df["days_elapsed"] + 1.0, alpha)

        # Aggregate
        trend_scores = temp_df.groupby("item_idx")["score"].sum().reset_index()

        # Normalize and Scale to [0, 9]
        max_val = trend_scores["score"].max()
        if max_val > 0:
            trend_scores["score"] = (trend_scores["score"] / max_val) * 9.0

        # Create dense vector
        trends = np.zeros(len(self.item_map), dtype=self.precision)
        trends[trend_scores["item_idx"].values] = trend_scores["score"].values

        # Cache
        os.makedirs(self.working_dir, exist_ok=True)
        trend_scores.to_parquet(cache_path, index=False)

        return trends

    def _get_user_history(self, train_df, n_users, n_items, load_cached_data):
        """
        Computes or loads the User History sparse matrix.
        Values = Sum(1 / (days_elapsed + 1))
        """
        cache_path = config.CACHE_USER_HISTORY

        if load_cached_data and os.path.exists(cache_path):
            print("Loading user history from cache...")
            try:
                return sp.load_npz(cache_path)
            except Exception as e:
                print(f"Error loading history cache: {e}. Recomputing...")

        print("Computing user history vectors...")
        if "days_elapsed" not in train_df.columns:
            max_date = train_df["t_dat"].max()
            train_df["days_elapsed"] = (max_date - train_df["t_dat"]).dt.days

        # Compute recency score
        # Note: No alpha here, or alpha=1.0 implicit for history as per "1/days" logic
        temp_df = train_df[["user_idx", "item_idx", "days_elapsed"]].copy()
        temp_df["score"] = 1.0 / (temp_df["days_elapsed"] + 1.0)

        # Aggregate duplicate purchases (summing scores rewards frequency AND recency)
        grouped = temp_df.groupby(["user_idx", "item_idx"])["score"].sum().reset_index()

        # Construct CSR Matrix
        row = grouped["user_idx"].values
        col = grouped["item_idx"].values
        data = grouped["score"].values.astype(self.precision)

        H = sp.csr_matrix(
            (data, (row, col)), shape=(n_users, n_items), dtype=self.precision
        )

        # Cache
        os.makedirs(self.working_dir, exist_ok=True)
        sp.save_npz(cache_path, H)

        return H

    def predict(self, target_users_df, k=12, batch_size=None):
        """
        Generates predictions for the provided users.

        Args:
            target_users_df (pd.DataFrame): Must contain 'customer_id'.
            k (int): Number of recommendations per user.
            batch_size (int): Batch size for processing.

        Returns:
            pd.DataFrame: DataFrame with 'customer_id' and 'prediction'.
        """
        if batch_size is None:
            batch_size = config.BATCH_SIZE

        # Filter for valid users (those in our user_map)
        # Cold start users totally new to the system (not in train/val/test union) are rare
        # but handled by the loader logic usually.
        valid_mask = target_users_df["customer_id"].isin(self.user_map.index)
        valid_users = target_users_df[valid_mask].copy()
        valid_users["user_idx"] = (
            valid_users["customer_id"].map(self.user_map).astype(int)
        )

        user_indices = valid_users["user_idx"].values
        customer_ids = valid_users["customer_id"].values

        n_total = len(user_indices)
        predictions = []

        print(
            f"Generating predictions for {n_total} users in batches of {batch_size}..."
        )

        # Prepare Global Trends (Broadcastable)
        # Shape: (1, n_items)
        trends_dense = self.global_trends.reshape(1, -1)

        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)
            batch_u_idxs = user_indices[start_idx:end_idx]
            current_bs = len(batch_u_idxs)

            # --- STRATUM 3: GLOBAL TRENDS ---
            # Initialize scores with global trends
            # We copy to a new array to accumulate other scores
            scores = np.empty((current_bs, len(self.item_map)), dtype=self.precision)
            scores[:] = trends_dense

            # Retrieve User History for this batch
            H_batch = self.user_history[batch_u_idxs]

            # --- STRATUM 2: HYBRID CF ---
            # R = U_history * S_hybrid
            if self.S_hybrid is not None:
                CF_raw = H_batch.dot(self.S_hybrid)

                if CF_raw.nnz > 0:
                    # Normalize row-wise (Max norm) so best match is 1.0
                    CF_norm = normalize(CF_raw, norm="max", axis=1)

                    # Scale to [10, 900] (approx)
                    # Formula: score * 800 + 10
                    # Since it's sparse, we operate on .data
                    CF_norm.data = (
                        CF_norm.data * config.CF_SCALING_FACTOR + config.SCORE_OFFSET_CF
                    )

                    # Add to dense scores
                    # Note: Adding sparse to dense is generally efficient in scipy/numpy
                    scores += CF_norm.toarray()

                    del CF_norm, CF_raw

            # --- STRATUM 1: HISTORY ---
            # Add history directly with high offset
            if H_batch.nnz > 0:
                H_copy = H_batch.copy()
                # Offset: [1000, inf)
                H_copy.data = H_copy.data + config.SCORE_OFFSET_HISTORY
                scores += H_copy.toarray()
                del H_copy

            # --- RETRIEVAL ---
            # Efficient Top-K using argpartition
            # We want largest scores. argpartition sorts such that the k-th element is in position.
            # We use -k to get the top k at the end of the array.
            top_k_indices = np.argpartition(scores, -k, axis=1)[:, -k:]

            # Sort the top K (argpartition does not guarantee order)
            rows = np.arange(current_bs)[:, None]
            top_k_scores = scores[rows, top_k_indices]

            # Argsort gives ascending, so we flip or negate
            sorted_args = np.argsort(-top_k_scores, axis=1)
            final_indices = top_k_indices[rows, sorted_args]

            # --- FORMATTING ---
            for i in range(current_bs):
                u_preds = []
                for item_idx in final_indices[i]:
                    # Map int idx -> article_id (int) -> string (0-padded)
                    art_id_int = self.reverse_item_map.get(item_idx)
                    if art_id_int is not None:
                        u_preds.append(f"{art_id_int:010d}")
                predictions.append(" ".join(u_preds))

            # Memory cleanup
            del scores, H_batch

        # Compile Results
        pred_df = pd.DataFrame({"customer_id": customer_ids, "prediction": predictions})

        return pred_df

    def run_submission(self):
        """
        End-to-end method to load data, run inference on test set, and save submission.
        """
        # 1. Load
        _, _, test_df = self.load_resources(load_cached_data=True)

        # 2. Predict
        submission_df = self.predict(test_df)

        # 3. Save
        print(f"Saving submission to {self.submission_path}...")
        submission_df.to_csv(self.submission_path, index=False)
        print("Submission saved successfully.")
