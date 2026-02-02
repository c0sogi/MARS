import os
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from library.config import Config
from library.data_utils import DataManager
from library.similarity_engine import SimilarityEngine
from library.trend_engine import TrendEngine


class SMDCRecommender:
    """
    Stratified Metadata-Densified Cascade (SMDC) Recommender.

    Implements a four-stage stratified retrieval system:
    1. Habitual Repurchase (Priors) - Rank [1000, inf)
    2. Densified Collaborative Filtering (Discovery) - Rank [100, 900]
    3. Cohort-Based Trends (Segmented) - Rank [10, 90]
    4. Global Trend (Universal) - Rank [0, 9]
    """

    def __init__(self):
        self.config = Config
        self.data_manager = DataManager()
        self.sim_engine = SimilarityEngine()
        self.trend_engine = TrendEngine()

        # Model Artifacts
        self.indexer = None
        self.U = None
        self.S_hybrid = None
        self.global_trends = None
        self.cohort_trends = None

        # Data
        self.customers_df = None
        self.train_df = None
        self.test_df = None

    def fit(self, validate=False, load_cached_data=True):
        """
        Loads data and constructs the similarity matrices and trend vectors.
        If validate is True, it also performs evaluation on the validation set.

        Args:
            validate (bool): If True, uses validation split and prints MAP@12.
            load_cached_data (bool): If True, attempts to load intermediate artifacts from disk.
        """
        print(f"Initializing SMDC Model (Validate={validate})...")

        # 1. Load Data
        # DataManager handles caching of the dataframes and indexer
        self.train_df, self.test_df, self.customers_df, articles_df, self.indexer = (
            self.data_manager.load_data(
                validate=validate, load_cached_data=load_cached_data
            )
        )

        # 2. Build Matrices (delegated to engines with caching)
        self.U = self.sim_engine.build_user_item_matrix(
            self.train_df, self.indexer, load_cached_data=load_cached_data
        )

        self.S_hybrid = self.sim_engine.build_hybrid_matrix(
            self.U, articles_df, self.indexer, load_cached_data=load_cached_data
        )

        # 3. Build Trends (delegated to engines with caching)
        self.global_trends = self.trend_engine.get_global_trends(
            self.train_df, self.indexer, load_cached_data=load_cached_data
        )

        self.cohort_trends = self.trend_engine.get_cohort_trends(
            self.train_df,
            self.customers_df,
            self.indexer,
            load_cached_data=load_cached_data,
        )

        print("Model artifacts constructed successfully.")

        # 4. Validation
        if validate:
            self._evaluate()

    def _evaluate(self):
        """
        Internal method to calculate MAP@12 on the validation set.
        """
        print("Starting Validation Evaluation...")

        # Prepare Ground Truth
        # test_df contains transactions [customer_id, article_id]
        # Group by customer to get list of purchased items
        gt_df = (
            self.test_df.groupby("customer_id")["article_id"].apply(list).reset_index()
        )

        # We only predict for customers in the test set
        unique_test_users = gt_df[["customer_id"]].copy()

        # Generate Predictions
        preds = self.predict(unique_test_users)

        # Calculate MAP@12
        print("Calculating MAP@12...")
        map_score = self._calculate_map12(preds, gt_df["article_id"].tolist())
        print(f"Validation MAP@12: {map_score}")

    def _calculate_map12(self, preds, ground_truth):
        """
        Computes Mean Average Precision @ 12.
        """
        metric = []
        for p_str, gt_list in zip(preds, ground_truth):
            # Ground truth is list of int/str article_ids
            # p_str is space-separated string
            if not gt_list:
                metric.append(0.0)
                continue

            # Convert GT to string for comparison (ensure zfill)
            gt_set = set(str(x).zfill(10) for x in gt_list)

            p_list = p_str.split()
            score = 0.0
            num_hits = 0.0

            for i, pred in enumerate(p_list):
                if i >= 12:
                    break
                if pred in gt_set:
                    num_hits += 1.0
                    score += num_hits / (i + 1.0)

            metric.append(score / min(len(gt_list), 12))

        return np.mean(metric)

    def predict(self, test_users_df, batch_size=2000):
        """
        Generates predictions using the Stratified Cascade logic.

        Args:
            test_users_df (pd.DataFrame): DataFrame containing 'customer_id' column.
            batch_size (int): Number of users to process at once.

        Returns:
            list: List of space-separated article ID strings.
        """
        all_users = test_users_df["customer_id"].values
        preds = []

        # Mappings
        user_to_idx = self.indexer.user_to_idx
        idx_to_item = self.indexer.idx_to_item

        # Customer Age Bin Map
        # Ensure fast lookup
        cust_age_map = dict(
            zip(self.customers_df["customer_id"], self.customers_df["age_bin"])
        )

        # Pre-compute user indices (vectorized lookup)
        # Use a default of -1 for unknown users
        user_indices = np.array([user_to_idx.get(u, -1) for u in all_users])

        n_items = self.S_hybrid.shape[1]

        # Constants
        SCORE_GLOBAL = self.config.SCORE_GLOBAL
        SCORE_COHORT = self.config.SCORE_COHORT
        SCORE_CF = self.config.SCORE_CF
        SCORE_REPURCHASE = self.config.SCORE_REPURCHASE
        SCALE_CF = self.config.SCALE_CF

        print(f"Predicting for {len(all_users)} users in batches of {batch_size}...")

        for i in range(0, len(all_users), batch_size):
            batch_users = all_users[i : i + batch_size]
            batch_u_indices = user_indices[i : i + batch_size]
            current_bs = len(batch_users)

            # --- 1. Global Trends (Base) ---
            # Shape: (batch_size, n_items)
            # Use tile to broadcast global trend vector
            batch_scores = np.tile(self.global_trends, (current_bs, 1))
            batch_scores += SCORE_GLOBAL

            # --- 2. Cohort Trends ---
            # Add specific trend vector based on user age
            for j, user_id in enumerate(batch_users):
                age_bin = cust_age_map.get(user_id, 0)  # Default to bin 0 if unknown

                # Get trend vector (default to 0 vector if bin missing from trends)
                # cohort_trends keys are integers
                c_vec = self.cohort_trends.get(age_bin)
                if c_vec is not None:
                    batch_scores[j] += c_vec + SCORE_COHORT
                else:
                    # Just add offset if no specific trend found (rare)
                    batch_scores[j] += SCORE_COHORT

            # --- 3. Collaborative Filtering (Discovery) ---
            # Filter for valid users (those with history)
            valid_mask = batch_u_indices != -1
            valid_indices = batch_u_indices[valid_mask]

            if len(valid_indices) > 0:
                # Get history slice
                U_batch = self.U[valid_indices]

                # Compute CF: U * S
                # Result is (n_valid_users, n_items)
                R_cf = U_batch.dot(self.S_hybrid)
                if sp.issparse(R_cf):
                    R_cf = R_cf.toarray()

                # Scale and Shift
                R_cf = R_cf * SCALE_CF + SCORE_CF

                # Add to batch scores
                full_rows = np.where(valid_mask)[0]
                batch_scores[full_rows] += R_cf

                # --- 4. Habitual Repurchase (Priors) ---
                # Add massive score to items in history
                # We reuse U_batch to find these items
                coo = U_batch.tocoo()
                # coo.row maps to index in valid_indices
                # We need index in batch_scores (full_rows)
                batch_rows = full_rows[coo.row]
                batch_cols = coo.col

                # Add offset
                batch_scores[batch_rows, batch_cols] += SCORE_REPURCHASE

            # --- 5. Retrieval ---
            k = 12
            # argpartition on negative scores to get top k indices
            if n_items >= k:
                top_k_idx = np.argpartition(-batch_scores, k, axis=1)[:, :k]
            else:
                top_k_idx = np.argsort(-batch_scores, axis=1)[:, :n_items]

            # Sort within top k
            rows = np.arange(current_bs)[:, None]
            top_k_scores = batch_scores[rows, top_k_idx]

            # Sort descending
            sort_ord = np.argsort(-top_k_scores, axis=1)
            final_idx = top_k_idx[rows, sort_ord]

            # Map to Strings
            for row_idx in final_idx:
                items = [str(idx_to_item[x]).zfill(10) for x in row_idx]
                preds.append(" ".join(items))

            if i % 10000 == 0:
                gc.collect()

        return preds

    def generate_submission(self):
        """
        Generates predictions for the test set defined in fit() and saves to CSV.
        """
        if self.test_df is None:
            raise RuntimeError("Model must be fit before generating submission.")

        print("Generating Submission...")

        # In submission mode, test_df is just customer_ids
        preds = self.predict(self.test_df)

        sub_df = pd.DataFrame(
            {"customer_id": self.test_df["customer_id"], "prediction": preds}
        )

        out_path = self.config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sub_df.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")
