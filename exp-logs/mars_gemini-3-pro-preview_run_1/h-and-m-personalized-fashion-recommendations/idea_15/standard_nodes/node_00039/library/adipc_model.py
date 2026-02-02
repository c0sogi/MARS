import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from library import config, utils, data_handler, sparse_engine


class ADIPCRecommender:
    """
    Asymmetric-Decay Inventory-Projected Cascade (ADIPC) Recommender.

    Implements a Three-Stage Stratified Retrieval System:
    1. Habitual Repurchase (Habit > 2000)
    2. Collaborative Filtering (100 < CF < 1000)
    3. Global Trend (0 < Trend < 10)

    Architecture uses asymmetric decay:
    - Gentle Decay (0.3) for Structure Learning (Item-Item Graph)
    - Aggressive Decay (1.0) for Intent Inference (User Query)
    """

    def __init__(self):
        self.data_handler = data_handler.DataHandler()
        self.sparse_engine = sparse_engine.SparseEngine()

        # Model Artifacts
        self.similarity_matrix = None
        self.global_trend = None
        self.inventory_mask = None

        # Mappings (loaded during fit/predict)
        self.item_map = None
        self.user_map = None

        # Cache Paths
        self.sim_matrix_path = os.path.join(
            config.WORKING_DIR, config.CACHE_SIMILARITY_MATRIX
        )
        self.trend_path = os.path.join(config.WORKING_DIR, config.CACHE_GLOBAL_TREND)
        self.mask_path = os.path.join(config.WORKING_DIR, config.CACHE_INVENTORY_MASK)

    def fit(self, mode="submission", load_cached_data=True):
        """
        Builds the Item-Item Similarity Graph and Global Trends.

        Args:
            mode (str): 'submission' (full data) or 'validation' (split data).
            load_cached_data (bool): Whether to attempt loading artifacts from disk.
        """
        print(f"Initializing ADIPC Fit process (Mode: {mode})...")

        # Load Dataset to get metadata and mappings
        dataset = self.data_handler.load_dataset(
            mode=mode, load_cached_data=load_cached_data
        )
        history_df = dataset["history_df"]
        cutoff_date = dataset["cutoff_date"]
        self.item_map = dataset["item_map"]
        self.user_map = dataset["user_map"]

        # Check for cached artifacts
        if (
            load_cached_data
            and os.path.exists(self.sim_matrix_path)
            and os.path.exists(self.trend_path)
            and os.path.exists(self.mask_path)
        ):
            print("Loading model artifacts from cache...")
            self.similarity_matrix = sp.load_npz(self.sim_matrix_path)
            self.global_trend = np.load(self.trend_path)
            self.inventory_mask = np.load(self.mask_path)
            return

        print("Building model artifacts from scratch...")

        # --- 1. Structure Learning (Item-Item Graph) ---
        # Strategy: Gentle Decay over 10 weeks to preserve long-tail signal
        structure_df = self.data_handler.get_structure_data(history_df, cutoff_date)

        print(f"Calculating Gentle Decay weights (Rate={config.GENTLE_DECAY_RATE})...")
        weights = utils.calculate_decay_weights(
            structure_df["t_dat"], cutoff_date, config.GENTLE_DECAY_RATE
        )

        print("Building Interaction Matrix...")
        num_users = len(self.user_map)
        num_items = len(self.item_map)
        interaction_matrix = self.sparse_engine.build_user_item_matrix(
            structure_df, weights, num_users, num_items
        )

        print("Applying IDF Weighting and Row Normalization...")
        interaction_matrix = self.sparse_engine.apply_idf_weighting(interaction_matrix)
        interaction_matrix = self.sparse_engine.normalize_rows(interaction_matrix)

        print(f"Computing Similarity Matrix (Top-{config.MAX_NEIGHBORS})...")
        self.similarity_matrix = self.sparse_engine.compute_item_similarity(
            interaction_matrix, top_k=config.MAX_NEIGHBORS
        )

        # --- 2. Global Trend (Fallback) ---
        # Strategy: Popularity in last 1 week, scaled to [0, 10]
        print("Calculating Global Trends...")
        trend_start = cutoff_date - pd.Timedelta(weeks=1)
        trend_df = history_df[history_df["t_dat"] > trend_start]

        pop_counts = trend_df["article_idx"].value_counts()
        self.global_trend = np.zeros(num_items, dtype=config.FLOAT_DTYPE)

        if not pop_counts.empty:
            indices = pop_counts.index.values
            counts = pop_counts.values.astype(config.FLOAT_DTYPE)
            self.global_trend[indices] = counts

            # Scale
            max_val = self.global_trend.max()
            if max_val > 0:
                self.global_trend = self.global_trend / max_val
                self.global_trend = (
                    self.global_trend
                    * (config.TREND_MAX_SCORE - config.TREND_MIN_SCORE)
                    + config.TREND_MIN_SCORE
                )

        # --- 3. Inventory Mask (Projection) ---
        # Strategy: Only allow items sold in last 1 week
        print("Creating Inventory Mask...")
        self.inventory_mask = self.data_handler.get_active_inventory(
            history_df, cutoff_date
        )

        # --- 4. Caching ---
        print("Saving artifacts to disk...")
        sp.save_npz(self.sim_matrix_path, self.similarity_matrix)
        np.save(self.trend_path, self.global_trend)
        np.save(self.mask_path, self.inventory_mask)

        print("Fit complete.")

    def predict(self, target_user_indices, history_df, cutoff_date, batch_size=2000):
        """
        Generates predictions using the Asymmetric-Decay Cascade.

        Args:
            target_user_indices (np.array): Indices of users to predict for.
            history_df (pd.DataFrame): Full history for query construction.
            cutoff_date (pd.Timestamp): Reference date for decay.
            batch_size (int): Number of users to process per batch.

        Returns:
            pd.DataFrame: Submission DataFrame with 'customer_id' and 'prediction'.
        """
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        print(f"Starting Prediction for {len(target_user_indices)} users...")
        num_items = len(self.item_map)

        # --- 1. Query Construction (Intent Inference) ---
        # Strategy: Use FULL history with Aggressive Decay (1.0)
        print("Filtering history for target users...")
        # Optimization: Filter history to only relevant users
        target_users_set = set(target_user_indices)
        user_history = history_df[history_df["user_idx"].isin(target_users_set)].copy()

        print(
            f"Calculating Aggressive Decay weights (Rate={config.AGGRESSIVE_DECAY_RATE})..."
        )
        weights = utils.calculate_decay_weights(
            user_history["t_dat"], cutoff_date, config.AGGRESSIVE_DECAY_RATE
        )

        print("Building User Query Matrix...")
        # We build the query matrix for ALL target users at once (sparse is memory efficient)
        # Note: We use the full user space dimensions for easy indexing
        num_total_users = len(self.user_map)
        query_matrix = self.sparse_engine.build_user_item_matrix(
            user_history, weights, num_total_users, num_items
        )

        # --- 2. Batch Inference ---
        predictions = []
        customer_ids = []

        # Lookup dictionaries
        idx_to_cust = self.user_map.set_index("user_idx")["customer_id"].to_dict()
        idx_to_article = self.item_map.set_index("article_idx")["article_id"].to_dict()

        total_users = len(target_user_indices)

        for start_idx in range(0, total_users, batch_size):
            end_idx = min(start_idx + batch_size, total_users)
            batch_users = target_user_indices[start_idx:end_idx]

            # A. Extract Batch Query
            batch_query = query_matrix[batch_users]

            # B. Stratum 2: Collaborative Filtering
            # R_cf = (U_agg @ S_gentle)
            batch_cf = batch_query @ self.similarity_matrix

            # Convert to dense for combination and masking
            # Shape: (Batch, Items)
            batch_scores = batch_cf.toarray()

            # Apply Inventory Mask (Projection)
            batch_scores = batch_scores * self.inventory_mask

            # Normalize CF Scores to [CF_MIN, CF_MAX] per user
            mins = batch_scores.min(axis=1, keepdims=True)
            maxs = batch_scores.max(axis=1, keepdims=True)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0  # Prevent div/0

            target_range = config.CF_MAX_SCORE - config.CF_MIN_SCORE
            batch_scores = (
                (batch_scores - mins) / ranges
            ) * target_range + config.CF_MIN_SCORE

            # If a user had no CF signal (max was 0), reset row to 0 so Trend dominates
            mask_no_signal = (maxs == 0).flatten()
            batch_scores[mask_no_signal, :] = 0

            # C. Stratum 3: Global Trend (Fallback)
            # Add trend to all rows (additive ensemble)
            batch_scores += self.global_trend

            # D. Stratum 1: Habitual Repurchase (Priors)
            # R_habit = U_agg + HABIT_OFFSET
            # We add the offset to items present in the query (history)
            coo = batch_query.tocoo()
            rows = coo.row
            cols = coo.col
            data = coo.data

            # Add habit scores (decayed weight + large offset)
            habit_scores = data + config.HABIT_OFFSET
            batch_scores[rows, cols] += habit_scores

            # E. Retrieval (Top-K)
            k = config.TOP_K

            # argpartition to find top k indices (unsorted)
            # Note: If num_items < k, this might fail, but dataset has 100k items
            top_k_partition = np.argpartition(batch_scores, -k, axis=1)[:, -k:]

            # Sort the top k
            rows_idx = np.arange(len(batch_users))[:, None]
            top_k_scores = batch_scores[rows_idx, top_k_partition]

            # argsort produces indices relative to the partition
            sorted_relative_idx = np.argsort(-top_k_scores, axis=1)

            # Map back to global item indices
            final_indices = top_k_partition[rows_idx, sorted_relative_idx]

            # F. Formatting
            for i, user_idx in enumerate(batch_users):
                cust_id = idx_to_cust[user_idx]
                item_indices = final_indices[i]

                # Convert to strings
                pred_items = [str(idx_to_article.get(idx, "")) for idx in item_indices]
                pred_str = " ".join(pred_items)

                predictions.append(pred_str)
                customer_ids.append(cust_id)

            if (start_idx // batch_size) % 10 == 0:
                print(f"Processed {end_idx}/{total_users} users...")

        submission_df = pd.DataFrame(
            {"customer_id": customer_ids, "prediction": predictions}
        )

        return submission_df

    def generate_submission(self):
        """
        Executes the full pipeline to generate submission.csv.
        """
        # 1. Fit Model (Submission Mode)
        self.fit(mode="submission", load_cached_data=True)

        # 2. Load Data for Prediction
        dataset = self.data_handler.load_dataset(
            mode="submission", load_cached_data=True
        )

        # 3. Predict
        submission_df = self.predict(
            target_user_indices=dataset["target_users"],
            history_df=dataset["history_df"],
            cutoff_date=dataset["cutoff_date"],
        )

        # 4. Save
        save_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        print(f"Saving submission to {save_path}...")
        submission_df.to_csv(save_path, index=False)
        print("Submission generation complete.")
