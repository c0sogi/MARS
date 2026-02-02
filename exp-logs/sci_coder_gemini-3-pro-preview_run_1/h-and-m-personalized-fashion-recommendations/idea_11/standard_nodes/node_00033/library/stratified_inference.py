import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import minmax_scale
import os
import gc
from library import config


class StratifiedRecommender:
    """
    Implements the Three-Stage Stratified Retrieval System.

    Hierarchy:
    1. Habit (Repurchase) - High Priority (Offset 2000+)
    2. Collaborative Filtering (Item-Item) - Medium Priority (Offset 100+)
    3. Global Trend - Low Priority (Fallback)
    """

    def __init__(self):
        self.user_to_idx = None
        self.idx_to_user = None
        self.item_to_idx = None
        self.idx_to_item = None

        self.habit_matrix = None
        self.interaction_matrix = None
        self.similarity_matrix = None
        self.global_trend = None

        self.n_users = 0
        self.n_items = 0

    def fit(
        self,
        transactions_df,
        interaction_matrix,
        similarity_matrix,
        user_to_idx,
        idx_to_user,
        item_to_idx,
        idx_to_item,
        load_cached_data=True,
    ):
        """
        Prepares the recommender by building/storing necessary matrices.

        Args:
            transactions_df (pd.DataFrame): Raw transaction history.
            interaction_matrix (sp.csr_matrix): Pre-computed decayed interaction matrix (for CF).
            similarity_matrix (sp.csr_matrix): Pre-computed item-item similarity matrix.
            user_to_idx, idx_to_user, item_to_idx, idx_to_item: Mappings.
            load_cached_data (bool): Whether to load internal matrices from cache.
        """
        print("Fitting StratifiedRecommender...")

        self.user_to_idx = user_to_idx
        self.idx_to_user = idx_to_user
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item

        self.n_users = len(user_to_idx)
        self.n_items = len(item_to_idx)

        # Store provided matrices for Stratum 2 (CF)
        self.interaction_matrix = interaction_matrix
        self.similarity_matrix = similarity_matrix

        # Build Stratum 1 (Habit) and Stratum 3 (Trend) data
        self._build_habit_and_trend(transactions_df, load_cached_data)

        print("Model fitted successfully.")

    def _build_habit_and_trend(self, transactions_df, load_cached_data):
        """
        Constructs the Habit Matrix (1/t decay) and Global Trend vector.
        """
        cache_dir = config.CACHE_DIR
        habit_path = os.path.join(cache_dir, "habit_matrix.npz")
        trend_path = os.path.join(cache_dir, "global_trend.npy")

        if (
            load_cached_data
            and os.path.exists(habit_path)
            and os.path.exists(trend_path)
        ):
            print("Loading cached Habit Matrix and Global Trend...")
            self.habit_matrix = sp.load_npz(habit_path)
            self.global_trend = np.load(trend_path)
            return

        print("Computing Habit Matrix (1/days decay) and Global Trend...")

        # 1. Prepare Data
        # Calculate days elapsed
        max_date = transactions_df["t_dat"].max()
        days_elapsed = (max_date - transactions_df["t_dat"]).dt.days.values.astype(
            np.float32
        )

        # Avoid division by zero (though 0 days elapsed is fine, we use 1/(days+1))
        # Logic: Recent = High score.
        # Formula: 1 / (days_elapsed + 1)
        decay_weights = 1.0 / (days_elapsed + 1.0)

        # Map indices
        u_indices = (
            transactions_df["customer_id"]
            .map(self.user_to_idx)
            .fillna(-1)
            .values.astype(np.int32)
        )
        i_indices = (
            transactions_df["article_id"]
            .map(self.item_to_idx)
            .fillna(-1)
            .values.astype(np.int32)
        )

        # Filter valid
        mask = (u_indices >= 0) & (i_indices >= 0)
        u_indices = u_indices[mask]
        i_indices = i_indices[mask]
        weights = decay_weights[mask]

        # 2. Build Habit Matrix (Sparse)
        # Sum weights for repeat purchases
        self.habit_matrix = sp.coo_matrix(
            (weights, (u_indices, i_indices)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        ).tocsr()

        # 3. Build Global Trend (Dense)
        # Sum of decay weights per item across all users
        # We can sum the columns of the habit matrix directly
        print("Computing Global Trend...")
        # Sum along axis 0 (users) -> resulting shape (1, n_items)
        trend_sum = np.array(self.habit_matrix.sum(axis=0)).flatten()

        # Normalize Trend to [0, 10]
        if trend_sum.max() > 0:
            self.global_trend = (trend_sum / trend_sum.max()) * 10.0
        else:
            self.global_trend = trend_sum

        # Ensure correct type
        self.global_trend = self.global_trend.astype(np.float32)

        # Save to cache
        print(f"Saving Habit Matrix and Global Trend to {cache_dir}...")
        os.makedirs(cache_dir, exist_ok=True)
        sp.save_npz(habit_path, self.habit_matrix)
        np.save(trend_path, self.global_trend)

    def predict(self, customer_ids_to_predict, batch_size=1000):
        """
        Generates predictions for the given list of customer_ids.

        Args:
            customer_ids_to_predict (list/array): List of customer_id strings.
            batch_size (int): Number of users to process at once.

        Returns:
            pd.DataFrame: DataFrame with 'customer_id' and 'prediction' columns.
        """
        print(f"Generating predictions for {len(customer_ids_to_predict)} customers...")

        # Map requested customers to indices
        # If a customer is not in self.user_to_idx, they are effectively new/unknown
        # (though generate_mappings usually includes test users).
        # We handle this by assigning a special index or handling -1.

        req_u_indices = []
        valid_mask = []

        for cid in customer_ids_to_predict:
            if cid in self.user_to_idx:
                req_u_indices.append(self.user_to_idx[cid])
                valid_mask.append(True)
            else:
                # This should ideally not happen if mappings are generated correctly
                req_u_indices.append(-1)
                valid_mask.append(False)

        req_u_indices = np.array(req_u_indices, dtype=np.int32)
        n_predict = len(req_u_indices)

        # Result container
        final_predictions = []

        # Process in batches
        for start_idx in range(0, n_predict, batch_size):
            end_idx = min(start_idx + batch_size, n_predict)

            if start_idx % 10000 == 0:
                print(f"Processed {start_idx}/{n_predict} users...")

            batch_u_indices = req_u_indices[start_idx:end_idx]
            batch_size_actual = len(batch_u_indices)

            # Initialize scores with Global Trend (Stratum 3)
            # Shape: (Batch, Items)
            # We broadcast the 1D global trend to the batch
            batch_scores = np.tile(self.global_trend, (batch_size_actual, 1))

            # Identify valid users (known in training data)
            # If user is -1 (unknown), they only get Trend scores.
            known_user_mask = batch_u_indices >= 0
            known_u_indices = batch_u_indices[known_user_mask]

            if len(known_u_indices) > 0:
                # --- Stratum 2: Collaborative Filtering ---
                # R_cf = U_decayed * S
                # U_decayed is from self.interaction_matrix

                # Slicing CSR is efficient
                u_vecs = self.interaction_matrix[known_u_indices]

                # Compute raw CF scores
                cf_scores_sparse = u_vecs.dot(self.similarity_matrix)

                # Normalize CF scores to [0, 1] roughly to ensure offset works cleanly
                # Since doing row-wise min-max on sparse is hard, we rely on the fact
                # that L2 norm * Cosine Sim is usually < 1.0.
                # We just apply the offset directly to non-zeros.

                # Apply Offset
                cf_scores_sparse.data += config.OFFSET_CF

                # Add to batch scores
                # We need to map back to the batch rows.
                # cf_scores_sparse corresponds to known_u_indices.

                # Adding sparse to dense:
                # We iterate or use specialized addition.
                # For a batch of 1000, iterating is acceptable in Python or using dense conversion.
                # Converting 1000x100k to dense is 400MB. Fine.

                cf_dense = cf_scores_sparse.toarray()
                batch_scores[known_user_mask] += cf_dense

                # --- Stratum 1: Habit ---
                # R_habit = Habit Matrix lookup
                habit_vecs = self.habit_matrix[known_u_indices]

                # Apply Offset
                habit_vecs.data += config.OFFSET_HABIT

                habit_dense = habit_vecs.toarray()
                batch_scores[known_user_mask] += habit_dense

                # Clean up
                del cf_scores_sparse, cf_dense, habit_vecs, habit_dense, u_vecs

            # --- Retrieval ---
            # Select Top-K
            top_k = config.TOP_K

            # argpartition is faster than sort
            # We want indices of the largest k elements
            # argpartition puts k-th element in position, smaller before, larger after
            # So we take the last k

            # Note: argpartition does not sort the top k. We need to sort them afterwards.

            # Partition
            part_indices = np.argpartition(batch_scores, -top_k, axis=1)[:, -top_k:]

            # Get values to sort
            # Advanced indexing: row indices [0..B], col indices [part_indices]
            row_indices = np.arange(batch_size_actual)[:, None]
            top_values = batch_scores[row_indices, part_indices]

            # Sort indices (descending) within the top k
            # argsort gives indices relative to the subarray
            sort_indices = np.argsort(-top_values, axis=1)

            # Map back to original item indices
            final_indices = part_indices[row_indices, sort_indices]

            # Convert to strings
            for i in range(batch_size_actual):
                items = [self.idx_to_item[idx] for idx in final_indices[i]]
                pred_str = " ".join(map(str, items))
                final_predictions.append(pred_str)

            del batch_scores
            # gc.collect() # Optional, can slow down if called too often

        # Create DataFrame
        submission_df = pd.DataFrame(
            {"customer_id": customer_ids_to_predict, "prediction": final_predictions}
        )

        return submission_df
