import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import logging
import gc
from typing import List, Tuple
from library.config import Config
from library.graph_builder import GraphBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MSGRecommender:
    """
    Multi-Scale Stratified Graph Recommender.
    Implements the inference logic for the MSG-Cascade architecture.
    """

    def __init__(self, config: Config):
        self.config = config
        self.graph_builder = GraphBuilder(config)

    def _prepare_habit_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'weight_habit' column to dataframe for Habit graph construction.
        Weight = 1 / (days_elapsed + 1)
        """
        # Ensure date format
        if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
            df["t_dat"] = pd.to_datetime(df["t_dat"])

        max_date = df["t_dat"].max()
        days_elapsed = (max_date - df["t_dat"]).dt.days

        # Avoid division by zero and ensure strict decay
        # Using float32 for memory efficiency
        df["weight_habit"] = (1.0 / (days_elapsed + 1.0)).astype(np.float32)
        return df

    def _compute_global_trend(self, df: pd.DataFrame, item_map: dict) -> np.ndarray:
        """
        Computes the global trend vector based on 'weight_fast'.
        Returns a dense vector of shape (n_items,).
        """
        logger.info("Computing global trend vector...")

        # Aggregate weight_fast by article_id
        trend_series = df.groupby("article_id")["weight_fast"].sum()

        n_items = len(item_map)
        trend_vec = np.zeros(n_items, dtype=np.float32)

        # Map to indices
        for article_id, weight in trend_series.items():
            if article_id in item_map:
                idx = item_map[article_id]
                trend_vec[idx] = weight

        # Normalize immediately to [0, 1]
        max_val = trend_vec.max()
        if max_val > 0:
            trend_vec /= max_val

        return trend_vec

    def _normalize_and_scale_batch(
        self, scores: np.ndarray, offset: float, scale: float = 1000.0
    ) -> np.ndarray:
        """
        Row-wise normalization and scaling for a batch of scores.
        Formula: (Score / Max_Row_Score) * scale + offset
        """
        # Find max per row (user)
        # scores shape: (batch_size, n_items)
        row_maxs = scores.max(axis=1, keepdims=True)

        # Avoid division by zero
        row_maxs[row_maxs == 0] = 1.0

        # Normalize to [0, 1]
        scores_norm = scores / row_maxs

        # Scale and shift
        # Only apply to non-zero entries to preserve sparsity logic implicitly
        # (though we are working with dense arrays here)
        scores_final = scores_norm * scale + offset

        # Zero out entries that were originally zero (if strictly required)
        # However, for stratification, we want the offset to apply to valid signals.
        # If the original score was 0, it means no signal.
        # (0 / max) * scale + offset = offset. This would give 'offset' score to unobserved items.
        # We must mask zeros back to 0 to avoid recommending irrelevant items with high base scores.
        mask = scores > 0
        scores_final = scores_final * mask

        return scores_final

    def _get_batch_predictions(
        self,
        batch_indices: np.ndarray,
        X_fast: sp.csr_matrix,
        S_fast: sp.csr_matrix,
        X_slow: sp.csr_matrix,
        S_slow: sp.csr_matrix,
        X_habit: sp.csr_matrix,
        global_trend: np.ndarray,
    ) -> List[str]:
        """
        Computes predictions for a batch of users using the stratified cascade.
        """
        batch_size = len(batch_indices)

        # 1. Global Trend (Stratum 4)
        # Base score for all items: [0, 1000)
        # We start with this dense array
        # Shape: (batch_size, n_items)
        # Scale trend to [0, 1000)
        trend_score = global_trend * 1000.0
        # Broadcast to batch
        total_scores = np.tile(trend_score, (batch_size, 1))

        # 2. Slow CF (Stratum 3)
        # Range: [1000, 2000)
        # Extract user history subset
        u_slow = X_slow[batch_indices]
        # Compute raw scores
        r_slow = u_slow.dot(S_slow)
        if sp.issparse(r_slow):
            r_slow = r_slow.toarray()

        # Normalize and add to total
        if r_slow.sum() > 0:
            r_slow_strat = self._normalize_and_scale_batch(
                r_slow, self.config.SCORE_OFFSET_SLOW, scale=1000.0
            )
            # We take maximum to enforce strict layering or sum?
            # The idea is R_total = R_rep + R_fast + R_slow + R_trend
            # Since ranges are disjoint, Sum is equivalent to Layering if we assume
            # a lower layer doesn't exceed its bound.
            total_scores += r_slow_strat

        # 3. Fast CF (Stratum 2)
        # Range: [2000, 3000)
        u_fast = X_fast[batch_indices]
        r_fast = u_fast.dot(S_fast)
        if sp.issparse(r_fast):
            r_fast = r_fast.toarray()

        if r_fast.sum() > 0:
            r_fast_strat = self._normalize_and_scale_batch(
                r_fast, self.config.SCORE_OFFSET_FAST, scale=1000.0
            )
            total_scores += r_fast_strat

        # 4. Habit (Stratum 1)
        # Range: [3000, inf)
        # X_habit contains the raw 1/t weights
        u_habit = X_habit[batch_indices]
        if sp.issparse(u_habit):
            u_habit = u_habit.toarray()

        if u_habit.sum() > 0:
            r_habit_strat = self._normalize_and_scale_batch(
                u_habit, self.config.SCORE_OFFSET_HABIT, scale=1000.0
            )
            total_scores += r_habit_strat

        # 5. Select Top K
        # Use argpartition for efficiency
        k = self.config.TOP_K_PREDICT
        predictions = []

        for i in range(batch_size):
            scores = total_scores[i]
            # Get indices of top k
            # Note: argpartition puts top k at the end, not sorted
            if len(scores) > k:
                top_k_idx = np.argpartition(scores, -k)[-k:]
                # Sort these top k by score descending
                top_k_scores = scores[top_k_idx]
                sorted_idx = top_k_idx[np.argsort(-top_k_scores)]
            else:
                sorted_idx = np.argsort(-scores)

            # Map back to article_ids
            pred_ids = [
                self.graph_builder.reverse_item_map.get(idx, "") for idx in sorted_idx
            ]
            predictions.append(" ".join(pred_ids))

        return predictions

    def generate_submission(
        self,
        train_df: pd.DataFrame,
        test_customers_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        active_items: np.ndarray,
        load_cached: bool = True,
    ):
        """
        Main pipeline to generate predictions and save to submission file.
        """
        logger.info("Starting submission generation...")

        # 1. Initialize Mappings and Graphs
        # This will load from cache or build from scratch
        X_fast, S_fast, X_slow, S_slow = self.graph_builder.run(
            train_df,
            pd.read_csv(self.config.CUSTOMERS_PATH),  # Need full customers for mapping
            articles_df,
            active_items,
            load_cached=load_cached,
        )

        # 2. Build Habit Matrix
        # We need to compute this specifically as it's not standard in GraphBuilder
        habit_cache_path = os.path.join(self.config.WORKING_DIR, "X_habit.npz")
        X_habit = None

        if load_cached and os.path.exists(habit_cache_path):
            logger.info("Loading Habit Matrix from cache...")
            X_habit = sp.load_npz(habit_cache_path)
        else:
            logger.info("Building Habit Matrix...")
            train_df = self._prepare_habit_weights(train_df)
            # Use GraphBuilder's utility to build the sparse matrix
            X_habit = self.graph_builder.build_interaction_matrix(
                train_df, "weight_habit"
            )
            sp.save_npz(habit_cache_path, X_habit)
            logger.info("Saved Habit Matrix to cache.")

        # 3. Compute Global Trend
        global_trend = self._compute_global_trend(train_df, self.graph_builder.item_map)

        # 4. Prepare Test Users
        # Map customer_ids to indices
        test_uids = test_customers_df["customer_id"].values
        mapped_uids = []
        valid_indices = []  # Indices in the test_uids array

        # Identify which test users exist in our mapping
        # Users not in mapping (if any) are strictly cold start but GraphBuilder maps ALL customers
        # from customers.csv, so theoretically all should be mapped.
        for i, uid in enumerate(test_uids):
            if uid in self.graph_builder.user_map:
                mapped_uids.append(self.graph_builder.user_map[uid])
                valid_indices.append(i)
            else:
                # Should not happen if mappings are based on customers.csv
                logger.warning(f"User {uid} not found in mappings.")

        mapped_uids = np.array(mapped_uids)

        # 5. Batch Inference
        batch_size = 5000  # Adjust based on memory
        total_users = len(mapped_uids)
        all_preds = []

        logger.info(f"Predicting for {total_users} users in batches of {batch_size}...")

        for start_idx in range(0, total_users, batch_size):
            end_idx = min(start_idx + batch_size, total_users)
            batch_u_indices = mapped_uids[start_idx:end_idx]

            batch_preds = self._get_batch_predictions(
                batch_u_indices, X_fast, S_fast, X_slow, S_slow, X_habit, global_trend
            )
            all_preds.extend(batch_preds)

            if start_idx % (batch_size * 10) == 0:
                gc.collect()
                logger.info(f"Processed {end_idx}/{total_users} users.")

        # 6. Create Submission DataFrame
        # Handle potential missing users (though unlikely)
        final_preds = [""] * len(test_uids)
        for i, pred in zip(valid_indices, all_preds):
            final_preds[i] = pred

        submission_df = pd.DataFrame(
            {"customer_id": test_uids, "prediction": final_preds}
        )

        # 7. Save
        logger.info(f"Saving submission to {self.config.SUBMISSION_PATH}...")
        submission_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        logger.info("Submission generation complete.")
