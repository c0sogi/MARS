import os
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from library.config import BEHAVIOR_MATRIX_PATH, TOP_K_BEHAVIOR, SEED

# Set fixed seed for reproducibility
np.random.seed(SEED)


class CooccurrenceBuilder:
    """
    Constructs the Item-Item Behavioral Co-occurrence Matrix ($S_{behavior}$).

    This class transforms the User-Item history into an Item-Item similarity matrix
    using the formula S = (X_norm)^T @ X_norm, where X_norm is the IDF-weighted
    and L2-normalized user history matrix.
    """

    def __init__(self):
        pass

    def build_similarity_matrix(self, user_history_matrix, load_cached_data=True):
        """
        Computes the item-item similarity matrix.

        Args:
            user_history_matrix (sp.csr_matrix): Sparse matrix of shape (n_users, n_items).
                                                 Contains time-decayed interaction weights.
            load_cached_data (bool): If True, attempts to load the matrix from disk.

        Returns:
            sp.csr_matrix: Sparse similarity matrix of shape (n_items, n_items).
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(BEHAVIOR_MATRIX_PATH), exist_ok=True)

        # 1. Caching Logic
        if load_cached_data and os.path.exists(BEHAVIOR_MATRIX_PATH):
            print(
                f"Loading behavioral similarity matrix from {BEHAVIOR_MATRIX_PATH}..."
            )
            return sp.load_npz(BEHAVIOR_MATRIX_PATH)

        print("Computing behavioral similarity matrix from scratch...")

        # Work on a copy to preserve the input matrix
        # user_history_matrix is (Users x Items)
        X = user_history_matrix.copy()
        n_users, n_items = X.shape

        # --- 2. IDF Weighting ---
        print("Applying IDF weighting...")
        # Calculate Document Frequency (DF): Number of users who bought each item.
        # We treat any non-zero weight as an interaction.
        X_binary = X.copy()
        X_binary.data = np.ones_like(X_binary.data)

        # Sum columns to get DF (result is 1 x n_items matrix)
        doc_freq = np.array(X_binary.sum(axis=0)).flatten()

        # Calculate IDF: log(N / (df + 1))
        # Adding 1 to denominator for smoothing
        idf = np.log(n_users / (doc_freq + 1.0))

        # Apply IDF weights to columns of X
        # For CSR, X.indices contains the column index for each data point
        X.data *= idf[X.indices]

        # --- 3. User Normalization ---
        print("Applying User (Row) L2 normalization...")
        # Normalize rows to unit L2 norm.
        # This prevents users with many interactions from dominating the dot product.
        X = normalize(X, norm="l2", axis=1)

        # --- 4. Matrix Multiplication ---
        print(f"Computing X.T @ X (Shape: {n_items}x{n_items})...")
        # Compute cosine similarity between items
        # Result is (Items x Items)
        sim_matrix = X.T @ X

        # --- 5. Pruning ---
        print(f"Pruning to Top-{TOP_K_BEHAVIOR} per item...")
        sim_matrix = self._prune_matrix(sim_matrix, k=TOP_K_BEHAVIOR)

        # Save to cache
        print(f"Saving behavioral similarity matrix to {BEHAVIOR_MATRIX_PATH}...")
        sp.save_npz(BEHAVIOR_MATRIX_PATH, sim_matrix)

        return sim_matrix

    def _prune_matrix(self, matrix, k):
        """
        Retains only the top k values per row in a sparse matrix and removes the diagonal.

        Args:
            matrix (sp.csr_matrix): The matrix to prune.
            k (int): Number of top elements to keep.

        Returns:
            sp.csr_matrix: Pruned matrix.
        """
        # Ensure CSR format for efficient row slicing
        matrix = matrix.tocsr()

        new_data = []
        new_indices = []
        new_indptr = [0]

        n_rows = matrix.shape[0]

        # Iterate over rows to filter top-k
        for i in range(n_rows):
            row_start = matrix.indptr[i]
            row_end = matrix.indptr[i + 1]

            if row_start == row_end:
                new_indptr.append(new_indptr[-1])
                continue

            cols = matrix.indices[row_start:row_end]
            vals = matrix.data[row_start:row_end]

            # Remove self-similarity (diagonal)
            mask_diag = cols != i
            cols = cols[mask_diag]
            vals = vals[mask_diag]

            if len(vals) == 0:
                new_indptr.append(new_indptr[-1])
                continue

            # Keep top K
            if len(vals) > k:
                # np.argpartition puts the k largest elements at the end (unsorted)
                idx = np.argpartition(vals, -k)[-k:]
                cols = cols[idx]
                vals = vals[idx]

            new_data.append(vals)
            new_indices.append(cols)
            new_indptr.append(new_indptr[-1] + len(vals))

        # Flatten lists to arrays
        if new_data:
            new_data = np.concatenate(new_data)
            new_indices = np.concatenate(new_indices)
        else:
            new_data = np.array([])
            new_indices = np.array([])

        # Reconstruct CSR matrix
        pruned = sp.csr_matrix(
            (new_data, new_indices, new_indptr), shape=matrix.shape, dtype=np.float32
        )

        return pruned
