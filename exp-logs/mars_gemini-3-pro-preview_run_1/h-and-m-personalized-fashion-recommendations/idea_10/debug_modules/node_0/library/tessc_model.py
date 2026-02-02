import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from library.config import Config
from library.utils import Timer, reduce_mem_usage
from library.data_factory import DataFactory
from library.similarity_engine import SimilarityEngine


class TESSCRecommender:
    """
    Time-Embedded Similarity Stratified Cascade (TESSC) Recommender.

    Implements a three-stage stratified retrieval system:
    1. Habitual Repurchase (High Priority)
    2. Time-Embedded CF (Medium Priority)
    3. Global Trend (Low Priority)
    """

    def __init__(self):
        self.X = None  # Interaction Matrix
        self.S = None  # Similarity Matrix
        self.user_map = None
        self.item_map = None
        self.val_df = None

        # Inference artifacts
        self.idx_to_art = None
        self.cust_to_idx = None
        self.top_trend_items = None

    def fit(self, use_validation=False, load_cached_data=True):
        """
        Orchestrates the construction of the model artifacts.

        Args:
            use_validation (bool): If True, uses the validation split logic.
                                   If False, uses all data for final submission.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        print(f"[TESSC] Fitting model (Validation Mode: {use_validation})...")

        # 1. Load Interaction Matrix & Maps
        # DataFactory handles caching internally based on the file existence
        self.X, self.user_map, self.item_map, self.val_df = (
            DataFactory.get_interaction_matrix(
                is_validation=use_validation, load_cached_data=load_cached_data
            )
        )

        # 2. Compute Similarity Matrix
        # SimilarityEngine handles caching internally
        self.S = SimilarityEngine.compute_similarity(
            self.X, load_cached_data=load_cached_data
        )

        # 3. Compute Global Trends (Fallback Layer)
        with Timer("Global Trend Computation"):
            # Sum columns of X (which contains time-decay weights)
            # Result is a dense matrix of shape (1, n_items)
            trend_scores = np.array(self.X.sum(axis=0)).flatten()

            # Identify top items for fallback
            # We need enough to fill a user with 0 history/similarity
            # Sorting all items is fast enough (~100k items)
            top_trend_indices = np.argsort(-trend_scores)
            self.top_trend_items = top_trend_indices[: Config.TOP_K_PREDICT]

            print(f"[TESSC] Top Trend Item Indices: {self.top_trend_items[:5]}")

        # 4. Prepare Inference Maps
        with Timer("Inference Prep"):
            self.idx_to_art = self.item_map.set_index("item_idx")[
                "article_id"
            ].to_dict()
            self.cust_to_idx = self.user_map.set_index("customer_id")[
                "user_idx"
            ].to_dict()

        print("[TESSC] Fit complete.")

    def predict(self, customer_ids):
        """
        Generates predictions for a list of customer_ids using the stratified cascade.

        Args:
            customer_ids (list): List of customer_id strings.

        Returns:
            pd.DataFrame: DataFrame with columns ['customer_id', 'prediction']
        """
        batch_size = 5000
        predictions = []

        # Map customers to indices (-1 for cold start)
        user_indices = [self.cust_to_idx.get(c, -1) for c in customer_ids]
        n_users = len(user_indices)

        # Pre-compute trend string for cold-start users to save time
        trend_arts = [self.idx_to_art[i] for i in self.top_trend_items]
        trend_str = " ".join(map(str, trend_arts))

        print(f"[TESSC] Predicting for {n_users} users in batches of {batch_size}...")

        for i in range(0, n_users, batch_size):
            batch_indices = user_indices[i : i + batch_size]

            # Identify known users in this batch
            known_mask = np.array(batch_indices) != -1
            valid_batch_indices = np.array(batch_indices)[known_mask]

            batch_results = []

            if len(valid_batch_indices) > 0:
                # --- Vectorized Inference ---

                # 1. Retrieve History (Stratum 1)
                # Slicing CSR by rows is efficient
                U_batch = self.X[valid_batch_indices].astype(np.float32)

                # 2. Compute CF Scores (Stratum 2)
                # R_cf = U * S
                R_cf = U_batch.dot(self.S).astype(np.float32)

                # 3. Stratification & Aggregation

                # Stratum 1: Habit
                # Add offset to non-zero history items
                # We operate on a copy to avoid modifying the original X
                U_habit = U_batch.copy()
                U_habit.data += Config.STRATA_HABIT_OFFSET

                # Stratum 2: CF
                # Normalize R_cf row-wise to [0, 1] then scale
                # Compute row maxs
                row_maxs = np.array(R_cf.max(axis=1).todense()).flatten()
                row_maxs[row_maxs == 0] = 1.0  # Prevent division by zero

                # Create diagonal scaling matrix
                scale_factors = Config.STRATA_CF_SCALE / row_maxs
                scale_diag = sp.diags(scale_factors)

                # Apply scaling: R_cf = Scale * R_cf
                R_cf = scale_diag.dot(R_cf)

                # Add offset to non-zeros
                R_cf.data += Config.STRATA_CF_OFFSET

                # Combine: R_total = R_habit + R_cf
                # Since ranges are disjoint (Habit > 2000, CF < 1100), simple sum preserves order
                R_total = U_habit + R_cf

                # --- Retrieval ---

                # Convert to dense for argpartition
                # Shape: (Batch_Valid, N_Items) ~ 5000 x 100k floats ~ 2GB.
                # This fits in memory.
                R_dense = R_total.toarray()

                # Process each user in the valid batch
                for row_idx in range(len(valid_batch_indices)):
                    scores = R_dense[row_idx]

                    # Get top K indices
                    # argpartition moves top K to the end
                    k = Config.TOP_K_PREDICT
                    if len(scores) >= k:
                        best_indices = np.argpartition(scores, -k)[-k:]
                    else:
                        best_indices = np.arange(len(scores))

                    # Sort these top K by score descending
                    best_indices = best_indices[np.argsort(scores[best_indices])[::-1]]

                    # Filter out zero scores (items with no signal)
                    # Note: Trend items are not in R_total, so 0 means no history/sim
                    final_indices = [idx for idx in best_indices if scores[idx] > 0]

                    # Convert to Article IDs
                    pred_arts = [self.idx_to_art[idx] for idx in final_indices]

                    # Fill with Global Trend (Stratum 3)
                    if len(pred_arts) < k:
                        needed = k - len(pred_arts)
                        current_set = set(pred_arts)
                        for t_idx in self.top_trend_items:
                            t_art = self.idx_to_art[t_idx]
                            if t_art not in current_set:
                                pred_arts.append(t_art)
                                if len(pred_arts) == k:
                                    break

                    batch_results.append(" ".join(map(str, pred_arts)))

            # Re-assemble batch (interleaving known and unknown)
            final_batch_preds = []
            known_ptr = 0
            for is_known in known_mask:
                if is_known:
                    final_batch_preds.append(batch_results[known_ptr])
                    known_ptr += 1
                else:
                    # Cold start user -> Pure Trend
                    final_batch_preds.append(trend_str)

            predictions.extend(final_batch_preds)

            if (i + batch_size) % 100000 == 0:
                print(f"Processed {i + batch_size} users...")

        return pd.DataFrame({"customer_id": customer_ids, "prediction": predictions})

    def evaluate(self):
        """
        Calculates MAP@12 on the validation set.
        """
        if self.val_df is None:
            print("[TESSC] No validation data available. Skipping evaluation.")
            return

        print("[TESSC] Starting evaluation...")

        # 1. Prepare Ground Truth
        # Group by customer -> list of articles
        val_grouped = (
            self.val_df.groupby("customer_id")["article_id"].apply(list).reset_index()
        )
        val_customers = val_grouped["customer_id"].tolist()
        ground_truth = val_grouped["article_id"].tolist()

        # 2. Generate Predictions
        # We only predict for customers in the validation set
        pred_df = self.predict(val_customers)

        # 3. Calculate MAP@12
        # Parse prediction strings back to lists
        predictions = pred_df["prediction"].str.split().tolist()
        # Convert strings to ints for comparison
        predictions = [[int(x) for x in p] for p in predictions]

        score = self._map_at_k(ground_truth, predictions, k=12)
        print(f"[TESSC] Validation MAP@12: {score:.10f}")
        return score

    def _map_at_k(self, true_items, pred_items, k=12):
        """
        Computes Mean Average Precision @ K
        """
        scores = []
        for true, pred in zip(true_items, pred_items):
            if not true:
                continue

            # Truncate predictions
            pred = pred[:k]

            score = 0.0
            num_hits = 0

            # Convert true to set for O(1) lookup
            true_set = set(true)

            for i, p in enumerate(pred):
                if p in true_set:
                    num_hits += 1
                    score += num_hits / (i + 1.0)

            scores.append(score / min(len(true), k))

        return np.mean(scores)

    def generate_submission(self):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("[TESSC] Generating submission...")

        # Load test customers
        test_df = pd.read_csv(Config.TEST_PATH)
        test_customers = test_df["customer_id"].tolist()

        # Predict
        submission_df = self.predict(test_customers)

        # Save
        save_path = Config.SUBMISSION_PATH
        submission_df.to_csv(save_path, index=False)
        print(f"[TESSC] Submission saved to {save_path}")
