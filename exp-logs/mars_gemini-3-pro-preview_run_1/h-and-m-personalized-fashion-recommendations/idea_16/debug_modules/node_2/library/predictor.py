import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import gc
from library import settings
from library.data_manager import TransactionLoader
from library.graph_model import InteractionGraph


class StratifiedRecommender:
    """
    Implements the Three-Stage Stratified Retrieval System (TWIG-SR).
    Generates predictions by aggregating disjoint score strata:
    1. Habitual Repurchase (Score > 2000)
    2. Time-Weighted CF (Score 100-1000)
    3. Global Trend (Score 0-10)
    """

    def __init__(self):
        self.working_dir = settings.WORKING_DIR
        self.submission_path = settings.PATH_SUBMISSION

        # Hyperparameters
        self.batch_size = 2000  # Number of users to process at once
        self.top_k = settings.TOP_K_PREDICTIONS

        # Stratification Offsets
        self.offset_habit = settings.HABIT_OFFSET
        self.offset_cf_min = settings.CF_OFFSET_MIN
        self.offset_cf_max = settings.CF_OFFSET_MAX
        self.scale_trend = settings.TREND_SCALE

    def _compute_global_trends(self, train_df, n_items, load_cached_data=True):
        """
        Computes the global popularity vector based on time-decayed sales velocity.
        """
        cache_path = settings.CACHE_GLOBAL_TRENDS

        if load_cached_data and os.path.exists(cache_path):
            print("[Predictor] Loading global trends from cache...")
            return np.load(cache_path)

        print("[Predictor] Computing global trends...")
        # Aggregate weights by item
        # train_df has 'item_idx' and 'weight' (which is time-decayed)
        trend_counts = train_df.groupby("item_idx")["weight"].sum()

        # Create dense vector
        global_trends = np.zeros(n_items, dtype=settings.FLOAT_DTYPE)
        global_trends[trend_counts.index] = trend_counts.values

        # Normalize to [0, TREND_SCALE]
        max_val = global_trends.max()
        if max_val > 0:
            global_trends = (global_trends / max_val) * self.scale_trend

        np.save(cache_path, global_trends)
        return global_trends

    def _build_habit_matrix(self, train_df, n_users, n_items):
        """
        Constructs a sparse matrix of habitual repurchase scores.
        Score = HABIT_OFFSET + (1 / (1 + days_elapsed))
        Keeps only the most recent purchase for each (user, item) pair to handle duplicates.
        """
        print("[Predictor] Building Habit Matrix...")

        # 1. Sort by days_elapsed ascending (most recent first)
        # This ensures that when we drop duplicates, we keep the most recent one
        df_sorted = train_df.sort_values("days_elapsed", ascending=True)

        # 2. Drop duplicates to keep only the most recent interaction per user-item pair
        df_dedup = df_sorted.drop_duplicates(
            subset=["user_idx", "item_idx"], keep="first"
        )

        # 3. Calculate Habit Scores
        # Formula: Offset + Recency Weight
        # days_elapsed is 0 for today. weight = 1/(1+0) = 1.
        recency_weight = 1.0 / (1.0 + df_dedup["days_elapsed"].values)
        habit_scores = self.offset_habit + recency_weight

        # 4. Build CSR Matrix
        row = df_dedup["user_idx"].values
        col = df_dedup["item_idx"].values

        H = sp.csr_matrix(
            (habit_scores, (row, col)),
            shape=(n_users, n_items),
            dtype=settings.FLOAT_DTYPE,
        )

        return H

    def run(self, validation=False):
        """
        Executes the full prediction pipeline.

        Parameters
        ----------
        validation : bool
            If True, runs on validation split and calculates MAP@12.
            If False, runs on full data and generates submission file.
        """
        # 1. Load Data
        loader = TransactionLoader()
        train_df, val_df, user_map, item_map = loader.get_data(validation=validation)

        n_users = len(user_map)
        n_items = len(item_map)

        # 2. Build/Load Graph Matrices (X, S)
        graph = InteractionGraph(n_users, n_items)
        graph.build(train_df)
        X, S = graph.get_matrices()

        # 3. Prepare Auxiliary Structures
        global_trends = self._compute_global_trends(train_df, n_items)
        H = self._build_habit_matrix(train_df, n_users, n_items)

        # 4. Identify Target Users
        if validation:
            # Predict only for users in the validation set
            target_user_indices = val_df["user_idx"].unique()
            print(
                f"[Predictor] Generating predictions for {len(target_user_indices)} validation users..."
            )
        else:
            # Predict for ALL users in the user map (which includes the submission file users)
            # We filter for those requested in the submission file later, but user_map covers them.
            # To be precise, we should iterate over the submission users.
            sub_df = pd.read_csv(settings.PATH_TEST)
            # Map submission customer_ids to indices
            target_user_indices = user_map[
                user_map["customer_id"].isin(sub_df["customer_id"])
            ]["user_idx"].values
            print(
                f"[Predictor] Generating predictions for {len(target_user_indices)} test users..."
            )

        # 5. Batch Inference
        # Pre-allocate arrays for mapping back
        # Reverse Item Map: item_idx -> article_id (str)
        idx_to_article = (
            item_map.set_index("item_idx")["article_id"].astype(str).to_dict()
        )

        predictions = []
        user_ids_processed = []

        # Process in batches to manage memory
        total_users = len(target_user_indices)

        for start_idx in range(0, total_users, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_users)
            batch_users = target_user_indices[start_idx:end_idx]
            current_batch_size = len(batch_users)

            # --- A. Initialize with Global Trends (Stratum 3) ---
            # Shape: (batch_size, n_items)
            # Broadcasting global_trends vector to matrix
            scores = np.tile(global_trends, (current_batch_size, 1))

            # --- B. Add Collaborative Filtering Scores (Stratum 2) ---
            # R_cf = X[batch] @ S
            # X is (n_users, n_items), S is (n_items, n_items)
            # We slice X to get (batch_size, n_items)
            X_batch = X[batch_users]

            # Sparse Matrix Multiplication
            # Result is (batch_size, n_items)
            R_cf_sparse = X_batch.dot(S)

            # Densify for normalization and addition
            # Note: This is the memory bottleneck. 2000 * 105k * 4 bytes ~= 840 MB. Safe.
            R_cf_dense = R_cf_sparse.toarray()

            # Row-wise Max Normalization
            # Avoid division by zero
            max_vals = R_cf_dense.max(axis=1, keepdims=True)
            max_vals[max_vals == 0] = 1.0  # Prevent div/0

            # Normalize to [0, 1]
            R_cf_norm = R_cf_dense / max_vals

            # Scale to [CF_OFFSET_MIN, CF_OFFSET_MAX]
            # Only apply to non-zero entries to preserve sparsity logic (though dense here)
            # Logic: If CF has signal, it overrides Trend.
            # Range: 100 + (0..1)*900 = [100, 1000]
            mask_cf = R_cf_dense > 0

            # Update scores
            # We use the mask to ensure we don't boost items with 0 CF score to 100
            scores[mask_cf] = self.offset_cf_min + (
                R_cf_norm[mask_cf] * (self.offset_cf_max - self.offset_cf_min)
            )

            # --- C. Add Habitual Repurchase Scores (Stratum 1) ---
            # H is (n_users, n_items). Slice for batch.
            H_batch = H[batch_users]

            # H contains values > 2000.
            # We can simply add them. Since H is sparse, we iterate or densify.
            # Densifying H_batch is safe (same size).
            H_dense = H_batch.toarray()

            # Where H has values, it overrides everything else because 2000 > 1000.
            # We can use maximum or addition.
            # Since strata are disjoint, max(current, habit) works perfectly.
            # Or just assignment where H > 0.
            mask_habit = H_dense > 0
            scores[mask_habit] = H_dense[mask_habit]

            # --- D. Retrieval (Top-K) ---
            # Use argpartition for O(n) selection of top k
            # We want indices of top 12 elements
            # argpartition puts the k-th largest at index -k, and larger ones after
            top_k_indices = np.argpartition(scores, -self.top_k, axis=1)[
                :, -self.top_k :
            ]

            # The result of argpartition is not sorted. We must sort the top k.
            # Get the values corresponding to these indices
            rows = np.arange(current_batch_size)[:, None]
            top_k_values = scores[rows, top_k_indices]

            # Sort indices based on values (descending)
            # argsort gives indices that sort the array. We want descending.
            sorted_local_indices = np.argsort(-top_k_values, axis=1)

            # Map back to global item indices
            final_indices = top_k_indices[rows, sorted_local_indices]

            # --- E. Format Predictions ---
            for i, u_idx in enumerate(batch_users):
                item_indices = final_indices[i]
                # Convert to string IDs
                pred_strings = [idx_to_article[idx] for idx in item_indices]

                # Format: "0706016001 0706016002 ..."
                # Ensure we pad with 0s if article_id was int
                # But map has them as strings already from the loader logic?
                # Loader keeps them as int. We converted `idx_to_article` to str above.
                # However, article_ids like 108775015 need to be '0108775015'.
                # Let's handle the zfill here to be safe.
                formatted_preds = ["0" + s if len(s) == 9 else s for s in pred_strings]

                predictions.append(" ".join(formatted_preds))
                user_ids_processed.append(u_idx)

            # Cleanup
            del scores, R_cf_sparse, R_cf_dense, H_dense

        # 6. Create Output DataFrame
        # Map user_idx back to customer_id
        idx_to_user = user_map.set_index("user_idx")["customer_id"].to_dict()
        customer_ids = [idx_to_user[uid] for uid in user_ids_processed]

        result_df = pd.DataFrame(
            {"customer_id": customer_ids, "prediction": predictions}
        )

        # 7. Validation or Submission
        if validation:
            from library.metrics import calculate_map_at_12

            print("[Predictor] Calculating MAP@12...")
            # Restore original IDs for metric calculation
            # val_df has ['user_idx', 'item_idx'], need ['customer_id', 'article_id']
            val_eval = val_df.merge(user_map, on="user_idx", how="left")
            val_eval = val_eval.merge(item_map, on="item_idx", how="left")

            score = calculate_map_at_12(val_eval, result_df)
            print(f"Validation MAP@12: {score:.10f}")
            return score
        else:
            print(f"[Predictor] Saving submission to {self.submission_path}...")
            # Ensure we include all customers from sample_submission
            sample_sub = pd.read_csv(settings.PATH_TEST)
            final_sub = sample_sub[["customer_id"]].merge(
                result_df, on="customer_id", how="left"
            )

            # Fill missing (if any) with global trend top 12
            # (Though our logic covers all mapped users, so this is just a safety net)
            if final_sub["prediction"].isnull().any():
                print("[Predictor] Filling missing predictions with global trends...")
                top_trend_idx = np.argsort(-global_trends)[:12]
                trend_str = " ".join(
                    [
                        (
                            "0" + str(idx_to_article[i])
                            if len(str(idx_to_article[i])) == 9
                            else str(idx_to_article[i])
                        )
                        for i in top_trend_idx
                    ]
                )
                final_sub["prediction"] = final_sub["prediction"].fillna(trend_str)

            final_sub.to_csv(self.submission_path, index=False)
            print("[Predictor] Submission saved successfully.")
