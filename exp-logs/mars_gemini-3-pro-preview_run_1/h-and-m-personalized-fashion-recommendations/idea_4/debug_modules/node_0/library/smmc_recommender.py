import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from library.config import (
    GLOBAL_TRENDS_PATH,
    SUBMISSION_PATH,
    STRATA_REP,
    STRATA_BEH,
    STRATA_VIS,
    STRATA_TREND,
    TOP_K_PREDICT,
    SEED,
)
from library.utils import format_submission

# Set fixed random seed
np.random.seed(SEED)


class GlobalTrendBuilder:
    """
    Computes and manages the global trend (popularity) vector.
    """

    def build_trends(self, transactions_df, mapper, load_cached_data=True):
        """
        Computes a dense vector of item popularity scores.

        Args:
            transactions_df (pd.DataFrame): Processed transactions with 'days_elapsed'.
            mapper (IndexMapper): Fitted mapper to align article IDs to matrix indices.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            np.ndarray: Dense array of shape (n_items,) with scores scaled to [0, 9].
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(GLOBAL_TRENDS_PATH), exist_ok=True)

        if load_cached_data and os.path.exists(GLOBAL_TRENDS_PATH):
            print(f"Loading global trends from {GLOBAL_TRENDS_PATH}...")
            df = pd.read_parquet(GLOBAL_TRENDS_PATH)
            # Ensure alignment with mapper is checked or assumed consistent by workflow
            # We reconstruct the array based on item_idx
            n_items = mapper.get_num_items()
            trends = np.zeros(n_items, dtype=np.float32)

            # Map values back
            # We assume the parquet contains 'item_idx' and 'score'
            if "item_idx" in df.columns and "score" in df.columns:
                indices = df["item_idx"].values
                scores = df["score"].values
                # Filter indices that might be out of bound if mapper changed (safety)
                valid = indices < n_items
                trends[indices[valid]] = scores[valid]
                return trends

        print("Computing global trends from scratch...")

        # Calculate weights: 1 / (days_elapsed + 1)
        # We use the same decay logic as user history
        df = transactions_df.copy()
        df["weight"] = 1.0 / (df["days_elapsed"] + 1.0)

        # Map article_id to item_idx
        df["item_idx"] = df["article_id"].map(mapper.item2idx)

        # Drop items not in the active mapper
        df = df.dropna(subset=["item_idx"])
        df["item_idx"] = df["item_idx"].astype(int)

        # Aggregate
        trend_series = df.groupby("item_idx")["weight"].sum()

        # Create dense vector
        n_items = mapper.get_num_items()
        trends = np.zeros(n_items, dtype=np.float32)
        trends[trend_series.index] = trend_series.values

        # Normalize to range [0, 9] (STRATA_TREND range)
        # We use Min-Max scaling
        min_val = trends.min()
        max_val = trends.max()

        if max_val > min_val:
            trends = (trends - min_val) / (max_val - min_val) * 9.0
        else:
            trends.fill(0.0)

        # Save to cache
        print(f"Saving global trends to {GLOBAL_TRENDS_PATH}...")
        # Save as dataframe for portability
        # Only save non-zero entries to save space
        non_zero_indices = np.nonzero(trends)[0]
        save_df = pd.DataFrame(
            {"item_idx": non_zero_indices, "score": trends[non_zero_indices]}
        )
        save_df.to_parquet(GLOBAL_TRENDS_PATH, index=False)

        return trends


class SMMCModel:
    """
    Stratified Multi-Modal Matrix Cascade Model.
    """

    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.trend_builder = GlobalTrendBuilder()

    def _scale_sparse_rows(self, matrix, target_min, target_max):
        """
        Scales non-zero values in each row of a sparse matrix to [target_min, target_max].

        Args:
            matrix (sp.csr_matrix): Input sparse matrix.
            target_min (float): Minimum value of the target range.
            target_max (float): Maximum value of the target range.

        Returns:
            sp.csr_matrix: Scaled matrix.
        """
        # Work on copy
        mat = matrix.copy()

        # Calculate max per row
        # max(axis=1) returns a matrix of shape (n_rows, 1)
        row_maxs = np.array(mat.max(axis=1).todense()).flatten()

        # Avoid division by zero
        row_maxs[row_maxs == 0] = 1.0

        # Create a scaling vector for data
        # We need to map each data element to its row index
        # CSR format: indptr points to row start/end

        # Efficient row scaling for CSR:
        # We repeat the row_max inverse for every non-zero element in that row
        row_indices = np.repeat(np.arange(mat.shape[0]), np.diff(mat.indptr))
        inv_maxs = 1.0 / row_maxs

        # Normalize to [0, 1] relative to the row's max
        mat.data *= inv_maxs[row_indices]

        # Scale to target range width: [0, range_width]
        range_width = target_max - target_min
        mat.data *= range_width

        # Add target_min to non-zero elements
        # This shifts the scores into the stratum
        mat.data += target_min

        return mat

    def predict(
        self,
        user_history,
        behavior_matrix,
        visual_matrix,
        mapper,
        transactions_df,
        load_cached_data=True,
    ):
        """
        Generates predictions using the stratified cascade.

        Args:
            user_history (sp.csr_matrix): User history matrix (Users x Items).
            behavior_matrix (sp.csr_matrix): Behavior similarity matrix (Items x Items).
            visual_matrix (sp.csr_matrix): Visual similarity matrix (Items x Items).
            mapper (IndexMapper): Mapper object.
            transactions_df (pd.DataFrame): For trend calculation.
            load_cached_data (bool): Cache flag.
        """
        print("Starting SMMC Inference...")

        # 1. Load Global Trends
        global_trends = self.trend_builder.build_trends(
            transactions_df, mapper, load_cached_data
        )

        n_users = user_history.shape[0]
        n_items = user_history.shape[1]

        # Prepare result storage
        all_preds = []
        all_customer_ids = []

        # Process in batches
        # We iterate through the user_history matrix
        for start_idx in tqdm(
            range(0, n_users, self.batch_size), desc="Predicting Batches"
        ):
            end_idx = min(start_idx + self.batch_size, n_users)
            batch_size_actual = end_idx - start_idx

            # Get batch history
            u_batch = user_history[start_idx:end_idx]

            # --- Stratum 4: Global Trend (Base) ---
            # Range: [0, 9]
            # Initialize dense score matrix with global trends
            # Broadcast trend vector to (batch_size, n_items)
            batch_scores = np.tile(global_trends, (batch_size_actual, 1))

            # --- Stratum 3: Visual ($V_{vis}$) ---
            # Range: [10, 90]
            # Compute raw scores: U @ S_vis
            vis_scores = u_batch.dot(visual_matrix)

            # Scale and Add
            if vis_scores.nnz > 0:
                vis_scores = self._scale_sparse_rows(vis_scores, 10.0, 90.0)
                # Add to dense accumulator
                # Convert sparse to dense and add.
                # Note: vis_scores is (batch, items). Usually sparse but can be dense if batch is large.
                # To save memory, we can iterate rows, but with 1000 batch size, dense add is fine.
                batch_scores += vis_scores.toarray()

            # --- Stratum 2: Behavior ($V_{beh}$) ---
            # Range: [100, 900]
            beh_scores = u_batch.dot(behavior_matrix)

            if beh_scores.nnz > 0:
                beh_scores = self._scale_sparse_rows(beh_scores, 100.0, 900.0)
                batch_scores += beh_scores.toarray()

            # --- Stratum 1: Repurchase ($V_{rep}$) ---
            # Range: [1000, inf)
            # Use u_batch directly.
            # u_batch weights are 1/(days+1), max is 1.0.
            # We just add 1000 to non-zeros.
            rep_scores = u_batch.copy()
            if rep_scores.nnz > 0:
                rep_scores.data += 1000.0
                batch_scores += rep_scores.toarray()

            # --- Retrieval ---
            # Select Top-K
            # argpartition is efficient for finding top k indices unsorted
            # We want top 12

            # Handle case where n_items < TOP_K_PREDICT (unlikely but possible in small debug sets)
            k = min(TOP_K_PREDICT, n_items)

            # argpartition puts the k largest elements at indices [-k:]
            top_k_indices = np.argpartition(batch_scores, -k, axis=1)[:, -k:]

            # The indices are unsorted. We need to sort them by score descending.
            # Extract the scores for the top k indices
            rows = np.arange(batch_size_actual)[:, None]
            top_k_scores = batch_scores[rows, top_k_indices]

            # Sort indices within the top k
            # argsort is ascending, so we reverse
            sort_order = np.argsort(top_k_scores, axis=1)[:, ::-1]

            # Apply sort order to indices
            final_indices = top_k_indices[rows, sort_order]

            # Map indices to Article IDs
            batch_preds = []
            for i in range(batch_size_actual):
                item_indices = final_indices[i]
                article_ids = [mapper.idx2item[idx] for idx in item_indices]
                batch_preds.append(article_ids)

            all_preds.extend(batch_preds)

            # Get Customer IDs for this batch
            # mapper.idx2user maps index to customer_id
            batch_cust_indices = range(start_idx, end_idx)
            batch_cust_ids = [mapper.idx2user[idx] for idx in batch_cust_indices]
            all_customer_ids.extend(batch_cust_ids)

        # --- Formatting & Saving ---
        print("Formatting submission...")
        submission_df = format_submission(all_customer_ids, all_preds)

        print(f"Saving submission to {SUBMISSION_PATH}...")
        submission_df.to_csv(SUBMISSION_PATH, index=False)

        print("SMMC Inference Complete.")
        return submission_df
