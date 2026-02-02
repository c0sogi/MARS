import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import pandas as pd
import os
from library import settings


class InteractionGraph:
    """
    Implements the Time-Weighted Interaction Graph (TWIG) model.
    Encapsulates sparse matrix construction, IDF weighting, normalization,
    and similarity computation with pruning.
    """

    def __init__(self, n_users, n_items):
        """
        Initialize the InteractionGraph.

        Parameters
        ----------
        n_users : int
            Total number of users in the system (including test users).
        n_items : int
            Total number of items in the system.
        """
        self.n_users = n_users
        self.n_items = n_items
        self.X = None  # User-Item Interaction Matrix (Weighted, Normalized)
        self.S = None  # Item-Item Similarity Matrix (Pruned)

    def build(self, train_df, load_cached_data=True):
        """
        Builds the interaction and similarity matrices from transaction data.
        Implements caching to avoid redundant computations.

        Parameters
        ----------
        train_df : pd.DataFrame
            DataFrame containing columns ['user_idx', 'item_idx', 'weight'].
        load_cached_data : bool
            If True, attempts to load matrices from disk.
        """
        path_X = settings.CACHE_INTERACTION_MATRIX
        path_S = settings.CACHE_SIMILARITY_MATRIX

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(path_X) and os.path.exists(path_S):
            print("[Graph] Loading matrices from cache...")
            try:
                self.X = sp.load_npz(path_X)
                self.S = sp.load_npz(path_S)
                print(f"[Graph] Loaded X: {self.X.shape}, S: {self.S.shape}")
                return
            except Exception as e:
                print(f"[Graph] Cache load failed: {e}. Recomputing...")

        # 2. Compute from Scratch
        print("[Graph] Building matrices from scratch...")

        # A. Construct Raw Interaction Matrix
        # Use sum_duplicates implicitly via coo_matrix -> csr_matrix conversion
        # to handle multiple purchases of the same item by the same user.
        print("[Graph] Constructing sparse matrix...")
        row = train_df["user_idx"].values
        col = train_df["item_idx"].values
        data = train_df["weight"].values.astype(settings.FLOAT_DTYPE)

        X_raw = sp.coo_matrix(
            (data, (row, col)),
            shape=(self.n_users, self.n_items),
            dtype=settings.FLOAT_DTYPE,
        ).tocsr()

        # B. Apply IDF Weighting (Column-wise)
        # Penalize globally popular items.
        # IDF_i = log(N_users / (1 + count_i))
        print("[Graph] Applying IDF weighting...")
        # Count non-zeros per column
        X_bool = X_raw.copy()
        X_bool.data = np.ones_like(X_bool.data)
        item_counts = np.array(X_bool.sum(axis=0)).flatten()

        # Compute IDF vector
        # Using log1p to ensure stability and positive weights
        idf = np.log1p(self.n_users / (1.0 + item_counts))
        idf = idf.astype(settings.FLOAT_DTYPE)

        # Apply as diagonal multiplication
        Diag_IDF = sp.diags(idf)
        X_weighted = X_raw.dot(Diag_IDF)

        # C. Row-wise L2 Normalization
        # Normalize user vectors to handle power-user bias
        print("[Graph] Normalizing rows (L2)...")
        self.X = normalize(X_weighted, norm="l2", axis=1)

        # D. Compute Similarity Matrix S = X^T @ X
        # This yields Item-Item similarity based on user interaction patterns.
        print("[Graph] Computing Similarity S = X.T @ X...")
        # Result is (n_items x n_items)
        S_dense_ish = self.X.T.dot(self.X)

        # E. Prune Similarity Matrix
        # Retain only top-K neighbors per item to optimize inference
        print(f"[Graph] Pruning Similarity Matrix (Top-{settings.TOP_K_SIMILAR})...")
        self.S = self._prune_matrix(S_dense_ish, k=settings.TOP_K_SIMILAR)

        # 3. Save to Cache
        print(f"[Graph] Saving matrices to {settings.WORKING_DIR}...")
        sp.save_npz(path_X, self.X)
        sp.save_npz(path_S, self.S)
        print("[Graph] Build complete.")

    def _prune_matrix(self, mat, k):
        """
        Retains only the top k elements per row in a sparse matrix.
        Optimized for memory efficiency using boolean masking.
        """
        mat = mat.tocsr()
        n_rows = mat.shape[0]

        # Create a boolean mask for data elements to keep
        mask = np.zeros(len(mat.data), dtype=bool)

        # New indptr array
        new_indptr = np.zeros(n_rows + 1, dtype=np.int32)
        current_ptr = 0

        # Iterate over rows to identify top-k elements
        # This loop is efficient enough for ~100k items in Python
        for i in range(n_rows):
            start = mat.indptr[i]
            end = mat.indptr[i + 1]
            length = end - start

            if length <= k:
                # Keep all
                mask[start:end] = True
                current_ptr += length
            else:
                # Find indices of top k elements
                row_slice = mat.data[start:end]
                # argpartition puts the k-th largest element in position -k
                # and all larger elements after it.
                top_k_local_indices = np.argpartition(row_slice, -k)[-k:]

                # Map back to global data indices
                global_indices = start + top_k_local_indices
                mask[global_indices] = True
                current_ptr += k

            new_indptr[i + 1] = current_ptr

        # Construct pruned matrix
        new_data = mat.data[mask]
        new_indices = mat.indices[mask]

        S_pruned = sp.csr_matrix(
            (new_data, new_indices, new_indptr), shape=mat.shape, dtype=mat.dtype
        )
        return S_pruned

    def get_matrices(self):
        """
        Returns the processed interaction matrix X and similarity matrix S.
        """
        if self.X is None or self.S is None:
            raise ValueError("Graph not built. Call build() first.")
        return self.X, self.S
