import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import gc
from library.config import Config
from library.utils import calculate_map12, format_submission
from library.data_processor import (
    load_and_filter_data,
    create_mappings,
    process_customer_cohorts,
)
from library.matrix_factory import MatrixFactory
from library.trend_analyzer import TrendAnalyzer


class StratifiedRecommender:
    """
    Implements the Stratified Directional-Cohort Cascade (SDCC) inference logic.
    Generates recommendations by combining four signal strata:
    1. Habitual Repurchase (History)
    2. Directional Behavioral CF (Hybrid Matrix)
    3. Cohort-Based Trends (Demographic)
    4. Global Trends (Universal)
    """

    def __init__(
        self,
        user_history_matrix,
        hybrid_matrix,
        cohort_trends,
        global_trends,
        user_cohort_map,
        user_to_idx,
        item_to_idx,
        idx_to_item,
    ):
        """
        Args:
            user_history_matrix (sp.csr_matrix): Time-decayed user history (U).
            hybrid_matrix (sp.csr_matrix): Hybrid similarity matrix (S_hybrid).
            cohort_trends (dict): Mapping of cohort_idx to sparse trend vectors.
            global_trends (np.ndarray): Dense global trend vector.
            user_cohort_map (np.ndarray): Array mapping user_idx to cohort_idx.
            user_to_idx (dict): Mapping customer_id -> user_idx.
            item_to_idx (dict): Mapping article_id -> item_idx.
            idx_to_item (dict): Mapping item_idx -> article_id.
        """
        self.U = user_history_matrix
        self.S = hybrid_matrix
        self.cohort_trends = cohort_trends
        self.global_trends = global_trends
        self.user_cohort_map = user_cohort_map
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.n_items = len(item_to_idx)

    def predict(self, customer_ids, batch_size=1000):
        """
        Generates top-k predictions for a list of customer IDs.

        Args:
            customer_ids (list): List of customer_id strings.
            batch_size (int): Number of users to process at once.

        Returns:
            np.ndarray: Matrix of shape (n_customers, k) containing predicted item indices.
        """
        # Convert customer IDs to indices
        # Users not in the map (shouldn't happen with correct mappings) get -1
        user_indices = [self.user_to_idx.get(uid, -1) for uid in customer_ids]
        user_indices = np.array(user_indices, dtype=np.int32)

        n_users = len(user_indices)
        predictions = np.zeros((n_users, Config.TOP_K), dtype=np.int32)

        # Process in batches
        for start_idx in range(0, n_users, batch_size):
            end_idx = min(start_idx + batch_size, n_users)
            batch_u_idxs = user_indices[start_idx:end_idx]
            curr_batch_size = len(batch_u_idxs)

            # Initialize scores with Global Trends (Stratum 4)
            # Range: [0, 9] + Offset 0
            # Shape: (batch_size, n_items)
            scores = np.tile(self.global_trends, (curr_batch_size, 1))
            scores += Config.OFFSET_GLOBAL

            # Handle valid users (those who exist in our mapping)
            valid_mask = batch_u_idxs != -1
            valid_u_idxs = batch_u_idxs[valid_mask]

            if len(valid_u_idxs) > 0:
                # --- Stratum 3: Cohort Trends ---
                # Range: [10, 90] (Base [0, 80] + Offset 10)
                batch_cohorts = self.user_cohort_map[valid_u_idxs]
                unique_cohorts = np.unique(batch_cohorts)

                for c_idx in unique_cohorts:
                    if c_idx in self.cohort_trends:
                        # Get sparse vector for cohort
                        c_vec = self.cohort_trends[c_idx]  # shape (1, n_items)
                        c_dense = c_vec.toarray().flatten()

                        # Identify items relevant to this cohort
                        c_mask = c_dense > 0

                        # Create additive vector: Value + Offset
                        add_vec = np.zeros(self.n_items, dtype=np.float32)
                        add_vec[c_mask] = c_dense[c_mask] + Config.OFFSET_COHORT

                        # Apply to all users in this cohort within the batch
                        # Map back to the batch rows
                        user_in_cohort_mask = batch_cohorts == c_idx

                        # We need to apply this to the 'scores' matrix rows corresponding to valid users
                        # First, find the indices in 'scores' (which is size curr_batch_size)
                        # valid_mask maps batch -> valid. user_in_cohort_mask maps valid -> cohort.

                        # Construct full batch mask
                        full_batch_indices = np.arange(curr_batch_size)[valid_mask]
                        target_rows = full_batch_indices[user_in_cohort_mask]

                        if len(target_rows) > 0:
                            scores[target_rows, :] += add_vec

                # --- Stratum 2: Directional CF ---
                # Range: [100, 900] (Base [0, 800] + Offset 100)
                # Compute R = U_batch * S_hybrid
                u_batch_vec = self.U[valid_u_idxs]
                cf_batch = u_batch_vec.dot(self.S)  # Sparse (n_valid, n_items)

                cf_dense = cf_batch.toarray()

                # Normalize row-wise (Max scaling)
                max_vals = cf_dense.max(axis=1, keepdims=True)
                max_vals[max_vals == 0] = 1.0  # Prevent div by zero
                cf_dense = cf_dense / max_vals

                # Scale to range width (800)
                cf_dense *= 800.0

                # Add Offset where score > 0
                cf_active_mask = cf_dense > 0
                cf_dense[cf_active_mask] += Config.OFFSET_CF

                # Add to scores
                scores[valid_mask] += cf_dense

                # --- Stratum 1: Habitual Repurchase ---
                # Range: [1000, inf) (Base weights + Offset 1000)
                # u_batch_vec contains the time-decayed history weights
                hist_dense = u_batch_vec.toarray()

                # Scale weights slightly to preserve relative recency order clearly
                hist_dense *= 100.0

                # Add Offset
                hist_active_mask = hist_dense > 0
                hist_dense[hist_active_mask] += Config.OFFSET_HISTORY

                # Add to scores
                scores[valid_mask] += hist_dense

            # --- Selection: Top K ---
            # Use argpartition for O(n) selection of top k elements
            # We want the indices of the largest k elements
            k = Config.TOP_K

            # argpartition puts the top k elements at the end, but not sorted
            top_k_unsorted = np.argpartition(scores, -k, axis=1)[:, -k:]

            # Extract the scores corresponding to these indices to sort them
            row_indices = np.arange(curr_batch_size)[:, None]
            top_k_scores = scores[row_indices, top_k_unsorted]

            # Sort indices descending
            sort_indices = np.argsort(-top_k_scores, axis=1)

            # Map back to original item indices
            final_indices = top_k_unsorted[row_indices, sort_indices]

            # Store
            predictions[start_idx:end_idx] = final_indices

        return predictions


def run_inference_pipeline(load_cached_data=True):
    """
    Executes the full SDCC inference pipeline:
    1. Loads and filters data.
    2. Builds/Loads necessary matrices and trends.
    3. Runs validation (MAP@12).
    4. Generates and saves submission.
    """
    print("Starting SDCC Inference Pipeline...")

    # 1. Load Data
    train_df, val_df, test_df = load_and_filter_data(load_cached_data=load_cached_data)

    # 2. Mappings
    user_to_idx, idx_to_user, item_to_idx, idx_to_item = create_mappings(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 3. Cohorts
    user_cohort_map = process_customer_cohorts(
        user_to_idx, load_cached_data=load_cached_data
    )

    # 4. Matrices
    # U: User History
    U = MatrixFactory.build_user_history_matrix(
        train_df, user_to_idx, item_to_idx, load_cached_data=load_cached_data
    )

    # S: Similarity Matrices
    S_sym = MatrixFactory.build_symmetric_similarity(
        train_df, user_to_idx, item_to_idx, load_cached_data=load_cached_data
    )
    S_fwd = MatrixFactory.build_transition_matrix(
        train_df, user_to_idx, item_to_idx, load_cached_data=load_cached_data
    )
    S_hybrid = MatrixFactory.get_hybrid_matrix(S_sym, S_fwd)

    # Free memory
    del S_sym, S_fwd
    gc.collect()

    # 5. Trends
    global_trends = TrendAnalyzer.compute_global_trends(
        train_df, item_to_idx, load_cached_data=load_cached_data
    )
    cohort_trends = TrendAnalyzer.compute_cohort_trends(
        train_df,
        user_cohort_map,
        user_to_idx,
        item_to_idx,
        load_cached_data=load_cached_data,
    )

    # 6. Initialize Recommender
    recommender = StratifiedRecommender(
        user_history_matrix=U,
        hybrid_matrix=S_hybrid,
        cohort_trends=cohort_trends,
        global_trends=global_trends,
        user_cohort_map=user_cohort_map,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
    )

    # 7. Validation
    print("\n--- Validation Phase ---")
    val_customers = val_df["customer_id"].unique()
    print(f"Predicting for {len(val_customers)} validation customers...")

    val_preds_matrix = recommender.predict(val_customers)

    # Convert matrix to dict for MAP calculation
    val_preds_dict = {}
    for i, cid in enumerate(val_customers):
        # Map indices back to article IDs
        pred_items = [idx_to_item[idx] for idx in val_preds_matrix[i]]
        val_preds_dict[cid] = pred_items

    calculate_map12(val_df, val_preds_dict)

    # Free memory before full inference
    del val_preds_dict, val_preds_matrix
    gc.collect()

    # 8. Submission
    print("\n--- Submission Phase ---")
    test_customers = test_df["customer_id"].unique()
    print(f"Predicting for {len(test_customers)} test customers...")

    test_preds_matrix = recommender.predict(test_customers)

    format_submission(test_preds_matrix, test_customers, idx_to_item)
    print("Pipeline completed successfully.")
