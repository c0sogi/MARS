import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import gc
from sklearn.preprocessing import normalize

from library.utils import Timer, set_seed, calculate_map12
from library.data_loader import TransactionLoader
from library.matrix_factory import SparseMatrixBuilder
from library.similarity_engine import ItemSimilarityModel


class StratifiedRecommender:
    """
    Implements the Decay-Weighted Behavioral Cascade (DWBC) model.
    Orchestrates data loading, matrix construction, and stratified inference.
    """

    def __init__(self, working_dir="./working/idea_8"):
        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

        # Initialize components
        self.loader = TransactionLoader(cache_dir=working_dir)
        self.matrix_builder = SparseMatrixBuilder(cache_dir=working_dir)
        self.sim_engine = ItemSimilarityModel(cache_dir=working_dir)

    def run(self, train_weeks=10, val_days=7, validation=True):
        """
        Executes the full pipeline: Load -> Build -> Train -> Predict.
        """
        print(f"[StratifiedRecommender] Starting run (Validation={validation})...")

        # 1. Load Data
        # We extend the training window to 10 weeks to capture sufficient history for the 'Habit' stratum
        train_df, val_df, test_customers = self.loader.load_transactions(
            train_weeks=train_weeks, val_days=val_days, validation=validation
        )

        # 2. Build Interaction Matrix X
        # This matrix contains decay weights, IDF, and is L2 normalized.
        X, user_map, item_map = self.matrix_builder.build(train_df, test_customers)

        # 3. Compute Similarity Matrix S
        # S = X^T * X (Pruned to top 100)
        S = self.sim_engine.compute_similarity(X, top_k=100)

        # 4. Compute Global Trend Vector T
        # T is the column-wise sum of the decay-weighted interaction matrix X.
        # This represents "Recency-Weighted Popularity".
        print("[StratifiedRecommender] Computing Global Trends...")
        with Timer("Global Trend Calc"):
            # Sum columns (result is matrix of shape 1 x n_items)
            global_trend = np.array(X.sum(axis=0)).flatten()

            # Scale to [0, 10] as per Stratum 3 requirements
            max_trend = global_trend.max()
            if max_trend > 0:
                global_trend = 10.0 * (global_trend / max_trend)

            # Ensure float32 for consistency
            global_trend = global_trend.astype(np.float32)

        # 5. Inference
        if validation:
            # Predict for validation users
            target_users = val_df["customer_id"].unique()
            print(f"[StratifiedRecommender] Validating on {len(target_users)} users...")

            preds = self._predict_stratified(
                X, S, global_trend, user_map, item_map, target_users
            )

            # Calculate MAP@12
            print("[StratifiedRecommender] Calculating MAP@12...")
            score = calculate_map12(preds, val_df)
            print(f"Validation MAP@12: {score:.10f}")

        else:
            # Predict for test users
            target_users = test_customers["customer_id"].unique()
            print(
                f"[StratifiedRecommender] Generating submission for {len(target_users)} users..."
            )

            preds = self._predict_stratified(
                X, S, global_trend, user_map, item_map, target_users
            )

            # Save Submission
            sub_dir = "./submission"
            os.makedirs(sub_dir, exist_ok=True)
            sub_path = os.path.join(sub_dir, "submission.csv")
            preds.to_csv(sub_path, index=False)
            print(f"[StratifiedRecommender] Submission saved to {sub_path}")

    def _predict_stratified(
        self, X, S, global_trend, user_map, item_map, target_users_list
    ):
        """
        Performs vectorized stratified inference.
        Score = (Habit * 1e6) + (CF * 1e3) + Trend
        """
        # Create reverse mapping: Index -> Article ID
        # item_map is Series: index=ArticleID, value=Index
        # We need an array where array[i] = ArticleID
        # Sort by index to ensure correct order
        sorted_map = pd.Series(
            item_map.index.values, index=item_map.values
        ).sort_index()
        index_to_article = sorted_map.values

        # Filter valid users (those who exist in the matrix)
        valid_users = [u for u in target_users_list if u in user_map]
        user_indices = user_map[valid_users].values

        n_users = len(user_indices)
        batch_size = 2000  # Process in chunks to manage memory
        all_preds = []

        print(f"  Processing {n_users} users in batches of {batch_size}...")

        with Timer("Batch Inference"):
            for start in range(0, n_users, batch_size):
                end = min(start + batch_size, n_users)
                batch_user_idx = user_indices[start:end]
                current_batch_size = len(batch_user_idx)

                # --- Stratum 1: Habitual Repurchase ---
                # Retrieve history from X.
                # Multiplier 1,000,000 ensures scores > 2000 (assuming min weight > 0.002)
                # and strictly dominates CF.
                X_batch = X[batch_user_idx]
                R_habit = X_batch * 1000000.0

                # --- Stratum 2: Collaborative Filtering ---
                # R_cf = X_batch * S
                # Multiplier 1,000 ensures scores are in [0, 1000] range approx.
                # (X is L2 norm, S is cosine sim, dot product usually < 1.0)
                R_cf = X_batch.dot(S)
                R_cf = R_cf * 1000.0

                # --- Aggregate Sparse Signals ---
                R_total = R_habit + R_cf

                # --- Stratum 3: Global Trend & Selection ---
                # We densify the batch to add the dense Global Trend vector.
                # With 220GB RAM, a dense batch of 2000 x 100k floats is ~800MB. Very safe.
                R_dense = R_total.toarray()

                # Add Global Trend (Broadcast)
                # Range [0, 10]. Only affects ranking if Habit and CF are weak/zero.
                R_dense += global_trend

                # Select Top 12
                # argpartition puts top k at the end (unsorted)
                k = 12
                # Handle case where n_items < k (unlikely but safe)
                k = min(k, R_dense.shape[1])

                top_k_idx = np.argpartition(R_dense, -k, axis=1)[:, -k:]

                # Sort the top k to get correct ranking
                row_idx = np.arange(current_batch_size)[:, None]
                top_k_vals = R_dense[row_idx, top_k_idx]

                # argsort produces indices of indices. Sort descending.
                sort_order = np.argsort(-top_k_vals, axis=1)
                final_indices = top_k_idx[row_idx, sort_order]

                # --- Decode & Format ---
                # Map indices to article IDs
                # Flatten for fast lookup
                flat_indices = final_indices.flatten()
                flat_articles = index_to_article[flat_indices]

                # Reshape back to (Batch, 12)
                batch_preds = flat_articles.reshape(current_batch_size, k)

                # Format as space-separated string with leading zeros
                # Vectorized string formatting is tricky, list comp is fast enough for batch
                batch_pred_strings = [
                    " ".join([f"{aid:010d}" for aid in row]) for row in batch_preds
                ]

                # Append to result
                batch_df = pd.DataFrame(
                    {
                        "customer_id": valid_users[start:end],
                        "prediction": batch_pred_strings,
                    }
                )
                all_preds.append(batch_df)

                # Explicit GC to prevent memory fragmentation
                if (start // batch_size) % 5 == 0:
                    gc.collect()

        # Concatenate all batches
        full_preds = pd.concat(all_preds, ignore_index=True)

        # Handle users who might have been missed (e.g. not in user_map)?
        # The matrix builder uses union of train+test, so all should be present.
        # But if target_users_list had IDs not in map (shouldn't happen), we'd miss them.
        # We fill missing users with pure global trend if necessary.

        if len(full_preds) < len(target_users_list):
            missing_users = set(target_users_list) - set(full_preds["customer_id"])
            if missing_users:
                print(
                    f"  Warning: {len(missing_users)} users had no mapping. Predicting Trend."
                )
                # Predict top 12 trend items for everyone
                top_trend_idx = np.argsort(-global_trend)[:12]
                top_trend_arts = index_to_article[top_trend_idx]
                trend_str = " ".join([f"{aid:010d}" for aid in top_trend_arts])

                missing_df = pd.DataFrame(
                    {"customer_id": list(missing_users), "prediction": trend_str}
                )
                full_preds = pd.concat([full_preds, missing_df], ignore_index=True)

        return full_preds
