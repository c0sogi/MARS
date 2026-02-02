import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import os
from library.config import Config


class SparseEngine:
    """
    Handles the construction of sparse interaction matrices and similarity matrices
    for the Decay-Weighted Stratified Cascade model.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.dtype = np.float32 if Config.PRECISION == "float32" else np.float64

    def build_decay_matrix(self, train_df, n_users, n_items, load_cached_data=True):
        """
        Constructs the Time-Decayed, IDF-Weighted User-Item Interaction Matrix.

        Args:
            train_df (pd.DataFrame): DataFrame with columns ['user_id', 'item_id', 'days_elapsed'].
            n_users (int): Total number of users (dimension 0).
            n_items (int): Total number of items (dimension 1).
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            scipy.sparse.csr_matrix: Normalized interaction matrix (Users x Items).
        """
        cache_path = os.path.join(self.cache_dir, "X_decay.npz")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached interaction matrix from {cache_path}...")
            return sp.load_npz(cache_path)

        print("Building decay-weighted interaction matrix...")

        # 1. Compute IDF Weights
        # IDF_i = log(Total Users / (Users who bought i + 1))
        # We count unique users per item in the training window
        print("Computing IDF weights...")
        item_user_counts = train_df.groupby("item_id")["user_id"].nunique()

        # Map counts to all items (fill missing with 0)
        counts_series = pd.Series(0, index=np.arange(n_items))
        counts_series.update(item_user_counts)
        counts = counts_series.values

        # IDF calculation
        idf = np.log(n_users / (counts + 1) + 1e-6).astype(self.dtype)

        # 2. Compute Interaction Weights
        # Weight = (1 / (days_elapsed + 1)^power) * IDF
        print("Computing temporal decay weights...")
        days = train_df["days_elapsed"].values.astype(self.dtype)
        decay_weights = 1.0 / np.power(days + 1.0, Config.CF_DECAY_POWER)

        # Get item indices for lookup
        item_indices = train_df["item_id"].values

        # Combine Decay and IDF
        # Note: If a user bought an item multiple times, we sum the weights (implicit in COO->CSR conversion)
        # or we could take max. Summing captures frequency + recency.
        interaction_values = decay_weights * idf[item_indices]

        # 3. Construct Sparse Matrix
        print(f"Constructing sparse matrix ({n_users}x{n_items})...")
        user_indices = train_df["user_id"].values

        X = sp.coo_matrix(
            (interaction_values, (user_indices, item_indices)),
            shape=(n_users, n_items),
            dtype=self.dtype,
        ).tocsr()

        # 4. Normalize Rows (L2)
        # This prevents power users from dominating the similarity calculation
        print("Normalizing rows (L2)...")
        X = normalize(X, norm="l2", axis=1)

        # Cache result
        print(f"Caching interaction matrix to {cache_path}...")
        sp.save_npz(cache_path, X)

        return X

    def compute_similarity(self, X, load_cached_data=True):
        """
        Computes the Item-Item Similarity Matrix (S = X^T @ X) with row-wise pruning.

        Args:
            X (scipy.sparse.csr_matrix): User-Item interaction matrix.
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            scipy.sparse.csr_matrix: Item-Item similarity matrix (Items x Items).
        """
        cache_path = os.path.join(self.cache_dir, "S_matrix.npz")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached similarity matrix from {cache_path}...")
            return sp.load_npz(cache_path)

        print("Computing similarity matrix (X^T @ X)...")
        # X is (U, I). X.T is (I, U). Result is (I, I).
        # With 220GB RAM, we can attempt direct sparse multiplication.
        # The result might be dense-ish, but we immediately prune it.

        # Transpose
        Xt = X.T

        # Matrix Multiplication
        # Note: This computes cosine similarity between item columns of X
        # (since X rows are normalized, this is actually a user-weighted cosine-like similarity)
        S_full = Xt.dot(X)

        print(f"Pruning similarity matrix to top-{Config.CF_NEIGHBORS} neighbors...")
        S_pruned = self._prune_matrix(S_full, k=Config.CF_NEIGHBORS)

        # Cache result
        print(f"Caching similarity matrix to {cache_path}...")
        sp.save_npz(cache_path, S_pruned)

        return S_pruned

    def _prune_matrix(self, S, k):
        """
        Retains only the top-k values per row in a sparse matrix.
        Uses block-wise processing to manage memory.
        """
        n_items = S.shape[0]
        chunk_size = 1000  # Process 1000 rows at a time

        new_rows = []
        new_cols = []
        new_data = []

        # Ensure S is CSR for efficient slicing
        S = S.tocsr()

        for start_idx in range(0, n_items, chunk_size):
            end_idx = min(start_idx + chunk_size, n_items)

            # Extract block and densify
            # Densifying small blocks is faster than working with pure sparse indexing for top-k
            block = S[start_idx:end_idx].toarray()
            m, n = block.shape

            if n <= k:
                # Keep all non-zero elements
                r, c = block.nonzero()
                v = block[r, c]
                new_rows.append(r + start_idx)
                new_cols.append(c)
                new_data.append(v)
            else:
                # Use argpartition to find top-k indices efficiently
                # We want indices of largest elements.
                # argpartition sorts smallest first, so we negate block.
                # axis=1 is per row
                top_k_indices = np.argpartition(-block, k, axis=1)[:, :k]

                # Create row indices for fancy indexing
                row_indices = np.arange(m)[:, None]

                # Extract values
                top_k_values = block[row_indices, top_k_indices]

                # Filter out zero values (if any top-k are zero, we don't store them)
                # This keeps the matrix sparse
                mask = top_k_values > 1e-9

                valid_rows = (row_indices + start_idx)[mask]
                valid_cols = top_k_indices[mask]
                valid_vals = top_k_values[mask]

                new_rows.append(valid_rows.flatten())
                new_cols.append(valid_cols.flatten())
                new_data.append(valid_vals.flatten())

        # Concatenate all lists
        if new_rows:
            all_rows = np.concatenate(new_rows)
            all_cols = np.concatenate(new_cols)
            all_data = np.concatenate(new_data)
        else:
            all_rows = np.array([], dtype=np.int32)
            all_cols = np.array([], dtype=np.int32)
            all_data = np.array([], dtype=self.dtype)

        # Reconstruct CSR matrix
        S_pruned = sp.csr_matrix(
            (all_data, (all_rows, all_cols)), shape=S.shape, dtype=self.dtype
        )

        return S_pruned
