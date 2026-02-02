import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import gc
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union

from library import config
from library import utils


class SparseGraphRetriever:
    """
    Stage 1: Vectorized Sparse Retrieval.
    Implements a graph-based candidate generation using time-decayed co-occurrences.
    """

    def __init__(self):
        self.top_k = config.TOP_K_RETRIEVAL
        self.decay_factor = config.TIME_DECAY_FACTOR
        self.alpha = config.HISTORY_WEIGHT_ALPHA
        self.working_dir = config.WORKING_DIR

        # Paths for caching
        self.path_matrix_t = self.working_dir / "transition_matrix.npz"
        self.path_matrix_r = self.working_dir / "user_history.npz"
        self.path_cust_map = self.working_dir / "customer_map.npy"
        self.path_art_map = self.working_dir / "article_map.npy"
        self.path_pop = self.working_dir / "global_popularity.npy"

        # In-memory artifacts
        self.transition_matrix: Optional[sp.csr_matrix] = None  # Item x Item
        self.user_history_matrix: Optional[sp.csr_matrix] = None  # User x Item
        self.customer_ids: Optional[np.ndarray] = (
            None  # Array of customer_id strings (index = internal id)
        )
        self.article_ids: Optional[np.ndarray] = (
            None  # Array of article_id ints (index = internal id)
        )
        self.global_popularity: Optional[np.ndarray] = None  # Top items by weight

        # Reverse mappers (built on load)
        self.cust_to_idx: Dict[str, int] = {}
        self.art_to_idx: Dict[int, int] = {}

    def fit(self, transactions_df: pd.DataFrame, load_cached_data: bool = True):
        """
        Builds the transition matrix and user history vectors.
        """
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # 1. Try Loading Cache
        if load_cached_data and self._check_cache_exists():
            print("[SparseGraphRetriever] Loading cached matrices...")
            try:
                self._load_cache()
                return
            except Exception as e:
                print(f"[SparseGraphRetriever] Cache load failed: {e}. Recomputing...")

        # 2. Compute from Scratch
        with utils.Timer("SparseGraphRetriever Fit"):
            print("[SparseGraphRetriever] Building matrices from scratch...")

            # Ensure DateTime
            if not np.issubdtype(transactions_df["t_dat"].dtype, np.datetime64):
                transactions_df["t_dat"] = pd.to_datetime(transactions_df["t_dat"])

            # --- A. Mappings ---
            # Create contiguous integer indices
            print("Encoding IDs...")
            unique_users = transactions_df["customer_id"].unique()
            unique_items = transactions_df["article_id"].unique()

            self.customer_ids = unique_users
            self.article_ids = unique_items

            # Build fast lookups
            self.cust_to_idx = {cid: i for i, cid in enumerate(unique_users)}
            self.art_to_idx = {aid: i for i, aid in enumerate(unique_items)}

            # Map DataFrame columns
            # Using map is faster than replace for large dataframes
            # We need to handle the case where transactions might have items not in unique_items
            # (unlikely if unique_items comes from the same df, but good practice)
            user_indices = (
                transactions_df["customer_id"].map(self.cust_to_idx).astype(np.int32)
            )
            item_indices = (
                transactions_df["article_id"].map(self.art_to_idx).astype(np.int32)
            )

            # --- B. Time Decay Weights ---
            print("Calculating time decay...")
            max_date = transactions_df["t_dat"].max()
            # Calculate days diff
            days_diff = (max_date - transactions_df["t_dat"]).dt.days.values
            # w = decay ^ days
            weights = np.power(self.decay_factor, days_diff).astype(np.float32)

            # --- C. Build User History Matrix (R) ---
            print("Constructing User-Item History Matrix (R)...")
            # Shape: (n_users, n_items)
            # Duplicate (user, item) pairs are summed automatically by csr_matrix constructor
            n_users = len(unique_users)
            n_items = len(unique_items)

            self.user_history_matrix = sp.csr_matrix(
                (weights, (user_indices, item_indices)),
                shape=(n_users, n_items),
                dtype=np.float32,
            )

            # --- D. Global Popularity ---
            # Sum of weights per item across all users
            print("Calculating Global Popularity...")
            item_scores = np.array(self.user_history_matrix.sum(axis=0)).flatten()
            # Get top K indices
            top_indices = np.argsort(item_scores)[::-1][: self.top_k]
            self.global_popularity = self.article_ids[top_indices]

            # --- E. Build Transition Matrix (T) ---
            print("Constructing Item-Item Transition Matrix (T = R.T @ R)...")
            # T represents weighted co-occurrence
            # We do NOT row normalize, to preserve magnitude (confidence)
            self.transition_matrix = self.user_history_matrix.T.dot(
                self.user_history_matrix
            )

            # Set diagonal to 0 (don't recommend item just because user bought it,
            # unless it's a repurchase which is handled by alpha term)
            self.transition_matrix.setdiag(0)
            self.transition_matrix.eliminate_zeros()

            # Optional: Prune T to keep it sparse if it gets too dense
            # For 100k items, full density is huge. But co-occurrence is naturally sparse.
            # We check density.
            density = self.transition_matrix.nnz / (n_items * n_items)
            print(f"Transition Matrix Density: {density:.6f}")

            # --- F. Save Cache ---
            print("Saving artifacts...")
            self._save_cache()

            # Cleanup
            del user_indices, item_indices, weights
            gc.collect()

    def predict(
        self,
        customer_ids: Union[List, np.ndarray],
        user_history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generates candidates for the given customers.

        Args:
            customer_ids: List of customer_id strings to predict for.
            user_history_df: Optional DataFrame containing history for these users.
                             If None, uses the history stored in self.user_history_matrix (from fit).
                             If provided, constructs a temporary query vector.

        Returns:
            DataFrame with columns [customer_id, article_id, score, rank]
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_targets = len(customer_ids)
        print(f"[SparseGraphRetriever] Generating candidates for {n_targets} users...")

        # 1. Build Query Matrix (Q)
        if user_history_df is not None:
            # Construct Q from provided history (e.g., for validation set users)
            # Map IDs using the fitted mappers
            # Filter history to only include known items
            valid_mask = user_history_df["article_id"].isin(self.art_to_idx)
            filtered_hist = user_history_df[valid_mask].copy()

            # We also need to handle new users in the history df if they aren't in cust_to_idx?
            # Actually, for the query matrix, we just need rows 0..n_targets-1 corresponding to customer_ids input.
            # We need to map customer_ids input to 0..N for the query matrix rows.
            target_map = {cid: i for i, cid in enumerate(customer_ids)}

            # Filter history for target users
            filtered_hist = filtered_hist[filtered_hist["customer_id"].isin(target_map)]

            q_rows = filtered_hist["customer_id"].map(target_map).values
            q_cols = filtered_hist["article_id"].map(self.art_to_idx).values

            # Calculate weights (decay)
            max_date = filtered_hist["t_dat"].max()
            days_diff = (max_date - filtered_hist["t_dat"]).dt.days.values
            q_data = np.power(self.decay_factor, days_diff).astype(np.float32)

            query_matrix = sp.csr_matrix(
                (q_data, (q_rows, q_cols)),
                shape=(n_targets, len(self.article_ids)),
                dtype=np.float32,
            )
        else:
            # Use internal history
            # Map requested customers to internal indices
            # Handle cold start (users not in training data)
            internal_indices = []
            valid_mask = []  # Boolean mask for customer_ids

            for cid in customer_ids:
                if cid in self.cust_to_idx:
                    internal_indices.append(self.cust_to_idx[cid])
                    valid_mask.append(True)
                else:
                    internal_indices.append(0)  # Placeholder, will be masked out
                    valid_mask.append(False)

            valid_mask = np.array(valid_mask)

            if valid_mask.any():
                # Slice the pre-computed R matrix
                # Note: Slicing CSR by rows is efficient
                subset_indices = np.array(internal_indices)[valid_mask]
                query_matrix_subset = self.user_history_matrix[subset_indices]

                # Reconstruct full query matrix (including empty rows for cold users)
                # This is a bit expensive. Better strategy:
                # Compute scores for valid users, then merge.
                pass
            else:
                query_matrix_subset = None

        # 2. Compute Scores
        # S = Q @ T + alpha * Q
        # We handle the "valid_mask" logic here to avoid creating massive empty sparse matrices

        results = []

        # We process in batches to manage memory
        batch_size = 1000

        # Prepare fallback candidates (Global Popularity)
        # Pre-map popularity to internal indices? No, we need article_ids at output.
        # But for filling scores, we need internal indices.
        pop_indices = [
            self.art_to_idx[aid]
            for aid in self.global_popularity
            if aid in self.art_to_idx
        ]

        # Loop over customers
        # To optimize, we use the matrix operation for valid users

        # Identify valid users (those who have history either in internal R or provided df)
        if user_history_df is None:
            # Indices in self.user_history_matrix
            target_indices = []
            is_cold = []
            for cid in customer_ids:
                if cid in self.cust_to_idx:
                    target_indices.append(self.cust_to_idx[cid])
                    is_cold.append(False)
                else:
                    target_indices.append(-1)
                    is_cold.append(True)

            # Get valid subset
            valid_indices = [i for i in target_indices if i != -1]
            if valid_indices:
                Q_valid = self.user_history_matrix[valid_indices]
            else:
                Q_valid = None
        else:
            # Q is already built for all targets (cold users have empty rows in Q)
            Q_valid = query_matrix
            is_cold = [False] * n_targets  # Logic handled by empty rows in Q

        # Compute Scores for valid portion
        if Q_valid is not None and Q_valid.nnz > 0:
            # Sparse Matrix Multiplication
            # S = Q . T + alpha * Q
            scores_valid = Q_valid.dot(self.transition_matrix)
            scores_valid += self.alpha * Q_valid
        else:
            scores_valid = None

        # 3. Extract Top K
        # Iterate and construct result

        valid_idx_counter = 0

        for i, cid in enumerate(customer_ids):
            # Check if user was cold (not in training set)
            # If user_history_df was provided, is_cold is False, but row might be empty

            user_scores = None

            if not is_cold[i]:
                # Get row from scores_valid
                if scores_valid is not None:
                    row = scores_valid[valid_idx_counter]
                    valid_idx_counter += 1

                    if row.nnz > 0:
                        user_scores = row

            # Extract candidates
            cands = []
            cand_scores = []

            if user_scores is not None:
                # Get top K from sparse row
                # row is 1 x n_items CSR
                _, idxs = row.nonzero()
                vals = row.data

                if len(idxs) > self.top_k:
                    # Argpartition to get top K
                    top_k_args = np.argpartition(vals, -self.top_k)[-self.top_k :]
                    best_idxs = idxs[top_k_args]
                    best_vals = vals[top_k_args]

                    # Sort descending
                    sort_order = np.argsort(best_vals)[::-1]
                    best_idxs = best_idxs[sort_order]
                    best_vals = best_vals[sort_order]
                else:
                    # Take all
                    sort_order = np.argsort(vals)[::-1]
                    best_idxs = idxs[sort_order]
                    best_vals = vals[sort_order]

                # Map back to article_ids
                cands = self.article_ids[best_idxs]
                cand_scores = best_vals

            # Fallback / Fill
            # If we have fewer than K candidates, fill with popularity
            if len(cands) < self.top_k:
                n_needed = self.top_k - len(cands)
                # Filter popularity to exclude already selected
                existing = set(cands)
                fill = []
                fill_scores = []
                for pop_item in self.global_popularity:
                    if pop_item not in existing:
                        fill.append(pop_item)
                        fill_scores.append(0.0)  # Zero score for fallback
                        if len(fill) == n_needed:
                            break

                cands = np.concatenate([cands, fill])
                cand_scores = np.concatenate([cand_scores, fill_scores])

            # Create rows
            # customer_id, article_id, score, rank
            for rank, (aid, score) in enumerate(zip(cands, cand_scores)):
                results.append(
                    {
                        "customer_id": cid,
                        "article_id": aid,
                        "score": float(score),
                        "rank": rank + 1,
                    }
                )

        return pd.DataFrame(results)

    def _check_cache_exists(self) -> bool:
        return (
            self.path_matrix_t.exists()
            and self.path_matrix_r.exists()
            and self.path_cust_map.exists()
            and self.path_art_map.exists()
            and self.path_pop.exists()
        )

    def _save_cache(self):
        print(f"Saving cache to {self.working_dir}...")
        sp.save_npz(self.path_matrix_t, self.transition_matrix)
        sp.save_npz(self.path_matrix_r, self.user_history_matrix)
        np.save(self.path_cust_map, self.customer_ids)
        np.save(self.path_art_map, self.article_ids)
        np.save(self.path_pop, self.global_popularity)

    def _load_cache(self):
        self.transition_matrix = sp.load_npz(self.path_matrix_t)
        self.user_history_matrix = sp.load_npz(self.path_matrix_r)
        self.customer_ids = np.load(self.path_cust_map, allow_pickle=True)
        self.article_ids = np.load(self.path_art_map, allow_pickle=True)
        self.global_popularity = np.load(self.path_pop, allow_pickle=True)

        # Rebuild dicts
        self.cust_to_idx = {cid: i for i, cid in enumerate(self.customer_ids)}
        self.art_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}
