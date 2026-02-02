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
        self.decay_exponent = config.TIME_DECAY_EXPONENT
        self.history_limit = config.HISTORY_ITEM_LIMIT
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
            # w = 1 / (1 + days)^exponent (Cite solution_lesson_node_00002)
            weights = (1.0 / (days_diff + 1) ** self.decay_exponent).astype(np.float32)

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
            if len(item_scores) > 0:
                top_indices = np.argsort(item_scores)[::-1][: self.top_k]
                self.global_popularity = self.article_ids[top_indices]
            else:
                self.global_popularity = np.array([])

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
            if n_items > 0:
                density = self.transition_matrix.nnz / (n_items * n_items)
                print(f"Transition Matrix Density: {density:.6f}")
            else:
                print("Warning: No items found. Transition matrix is empty.")

            # --- F. Save Cache ---
            print("Saving artifacts...")
            self._save_cache()

            # Cleanup
            del user_indices, item_indices, weights
            gc.collect()

    def _prune_query_matrix(self, Q: sp.csr_matrix) -> sp.csr_matrix:
        """
        Implements Importance-Based History Truncation (Cite solution_lesson_node_00012).
        Keeps only the top K weighted items per user in the query vector.
        """
        k = self.history_limit
        if k <= 0:
            return Q

        new_data = []
        new_indices = []
        new_indptr = [0]

        # Iterate over CSR rows directly
        for i in range(Q.shape[0]):
            start, end = Q.indptr[i], Q.indptr[i + 1]
            if start == end:
                new_indptr.append(new_indptr[-1])
                continue

            row_data = Q.data[start:end]
            row_indices = Q.indices[start:end]

            if len(row_data) > k:
                # Keep top K by weight
                top_k_idx = np.argpartition(row_data, -k)[-k:]
                row_data = row_data[top_k_idx]
                row_indices = row_indices[top_k_idx]

            new_data.extend(row_data)
            new_indices.extend(row_indices)
            new_indptr.append(len(new_data))

        return sp.csr_matrix((new_data, new_indices, new_indptr), shape=Q.shape)

    def predict(
        self,
        customer_ids: Union[List, np.ndarray],
        user_history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generates candidates for the given customers using direct sparse matrix operations.
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_targets = len(customer_ids)
        print(f"[SparseGraphRetriever] Generating candidates for {n_targets} users...")

        # 1. Build Query Matrix (Q)
        if user_history_df is not None:
            # Filter history to only include known items
            valid_mask = user_history_df["article_id"].isin(self.art_to_idx)
            filtered_hist = user_history_df[valid_mask].copy()

            target_map = {cid: i for i, cid in enumerate(customer_ids)}
            filtered_hist = filtered_hist[filtered_hist["customer_id"].isin(target_map)]

            q_rows = filtered_hist["customer_id"].map(target_map).values
            q_cols = filtered_hist["article_id"].map(self.art_to_idx).values

            # Power-law decay for query weights
            max_date = filtered_hist["t_dat"].max()
            days_diff = (max_date - filtered_hist["t_dat"]).dt.days.values
            q_data = (1.0 / (days_diff + 1) ** self.decay_exponent).astype(np.float32)

            Q = sp.csr_matrix(
                (q_data, (q_rows, q_cols)),
                shape=(n_targets, len(self.article_ids)),
                dtype=np.float32,
            )
            # Sum duplicates (multiple purchases of same item)
            Q.sum_duplicates()
        else:
            # Use internal history (R)
            # Map requested customers to internal indices
            internal_indices = []
            row_map = []  # maps Q row index to R row index

            for i, cid in enumerate(customer_ids):
                if cid in self.cust_to_idx:
                    internal_indices.append(self.cust_to_idx[cid])
                    row_map.append(i)

            if internal_indices:
                # Extract rows from R
                R_subset = self.user_history_matrix[internal_indices]

                # Create Q with correct shape (n_targets, n_items)
                # We construct it by creating a new CSR from R_subset data
                # but re-mapping row indices to 0..n_targets
                # This is slightly complex, so we just use vstack or simpler:
                # Create empty Q
                Q = sp.lil_matrix((n_targets, len(self.article_ids)), dtype=np.float32)
                # Fill valid rows (slow for large N, but acceptable for inference batches)
                # Optimization: Construct CSR directly

                # Get R_subset in COO to shift row indices
                R_coo = R_subset.tocoo()
                # R_coo.row goes 0..len(internal_indices)
                # We need to map these to the original indices in customer_ids (row_map)
                new_rows = np.array(row_map)[R_coo.row]

                Q = sp.csr_matrix(
                    (R_coo.data, (new_rows, R_coo.col)),
                    shape=(n_targets, len(self.article_ids)),
                    dtype=np.float32,
                )
            else:
                Q = sp.csr_matrix((n_targets, len(self.article_ids)), dtype=np.float32)

        # 2. Prune Query Matrix (Importance-Based Truncation)
        Q = self._prune_query_matrix(Q)

        # 3. Compute Scores: S = Q @ T + alpha * Q
        if Q.nnz > 0:
            S = Q.dot(self.transition_matrix)
            S += self.alpha * Q
        else:
            S = sp.csr_matrix((n_targets, len(self.article_ids)), dtype=np.float32)

        # 4. Extract Top K (Direct CSR Access - Cite solution_lesson_node_00016)
        results = []

        # Pre-calculate fallback
        fallback_ids = self.global_popularity[: self.top_k]
        fallback_scores = np.zeros(len(fallback_ids))

        for i in range(n_targets):
            cid = customer_ids[i]
            start, end = S.indptr[i], S.indptr[i + 1]

            cands = []
            cand_scores = []

            if start < end:
                row_data = S.data[start:end]
                row_inds = S.indices[start:end]

                if len(row_data) > self.top_k:
                    # Argpartition for top K
                    top_k_idx = np.argpartition(row_data, -self.top_k)[-self.top_k :]
                    best_data = row_data[top_k_idx]
                    best_inds = row_inds[top_k_idx]

                    # Sort descending
                    sort_order = np.argsort(best_data)[::-1]
                    cands = self.article_ids[best_inds[sort_order]]
                    cand_scores = best_data[sort_order]
                else:
                    # Take all
                    sort_order = np.argsort(row_data)[::-1]
                    cands = self.article_ids[row_inds[sort_order]]
                    cand_scores = row_data[sort_order]

            # Fallback
            if len(cands) < self.top_k:
                n_needed = self.top_k - len(cands)
                existing = set(cands)
                fill = [x for x in fallback_ids if x not in existing][:n_needed]
                fill_scores = np.zeros(len(fill))

                cands = np.concatenate([cands, fill])
                cand_scores = np.concatenate([cand_scores, fill_scores])

            # Store results
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
