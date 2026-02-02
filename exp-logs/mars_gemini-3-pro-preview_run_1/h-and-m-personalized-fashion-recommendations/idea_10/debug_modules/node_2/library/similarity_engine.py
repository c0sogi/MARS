import numpy as np
import scipy.sparse as sp
import os
import pandas as pd
from sklearn.preprocessing import normalize
from library.config import Config
from library.utils import Timer


class SimilarityEngine:
    """
    Encapsulates the logic for the 'Time-Embedded Collaborative Filtering' stratum.
    Computes the Item-Item similarity matrix using vectorized sparse linear algebra.
    """

    @staticmethod
    def compute_similarity(interaction_matrix, load_cached_data=True):
        """
        Computes the Item-Item similarity matrix S = X^T * X.

        Applies:
        1. IDF Weighting to columns (Items).
        2. L2 Normalization to rows (Users).
        3. Dot product.
        4. Top-K Pruning per item.

        Args:
            interaction_matrix (sp.csr_matrix): The user-item interaction matrix (X).
            load_cached_data (bool): If True, attempts to load the matrix from disk.

        Returns:
            sp.csr_matrix: The pruned item-item similarity matrix (S).
        """
        # Define cache path
        cache_path = os.path.join(Config.WORKING_DIR, "similarity_matrix.npz")

        # 1. Attempt to load from cache
        if load_cached_data:
            if os.path.exists(cache_path):
                print(
                    f"[SimilarityEngine] Loading cached similarity matrix from {cache_path}..."
                )
                try:
                    S = sp.load_npz(cache_path)
                    return S
                except Exception as e:
                    print(
                        f"[SimilarityEngine] Error loading cache: {e}. Recomputing..."
                    )
            else:
                print(
                    f"[SimilarityEngine] Cache not found at {cache_path}. Computing..."
                )

        # 2. Compute from scratch
        with Timer("Similarity Computation"):
            # Ensure input is float32 for precision/memory balance
            X = interaction_matrix.astype(np.float32)

            # A. IDF Weighting
            if Config.IDF_WEIGHTING:
                print("[SimilarityEngine] Calculating and applying IDF weights...")
                # Number of users
                N = X.shape[0]

                # Count non-zeros per column (items)
                # Converting to CSC makes column operations efficient
                X_csc = X.tocsc()
                col_nnz = np.diff(X_csc.indptr)

                # Compute IDF: log(N / (df + 1))
                # Add 1 to df to avoid division by zero
                idf = np.log(N / (col_nnz + 1.0))
                idf = idf.astype(np.float32)

                # Apply weighting: X_weighted = X @ Diag(IDF)
                # sp.diags creates a sparse diagonal matrix
                D_idf = sp.diags(idf)
                X = X @ D_idf

            # B. L2 Normalization (Rows/Users)
            if Config.USER_L2_NORM:
                print("[SimilarityEngine] Applying L2 row normalization...")
                # normalize returns a copy or modifies in place depending on copy arg
                # We use sklearn's optimized sparse implementation
                X = normalize(X, norm="l2", axis=1)

            # C. Compute Similarity (S = X^T @ X)
            print(
                f"[SimilarityEngine] Computing sparse dot product X^T @ X. Input shape: {X.shape}..."
            )
            # Result is Item x Item
            S = X.T @ X

            # D. Zero Diagonal
            # We want to find neighbors, not the item itself.
            print("[SimilarityEngine] Zeroing diagonal...")
            S.setdiag(0)
            S.eliminate_zeros()

            # E. Pruning (Top-K)
            print(
                f"[SimilarityEngine] Pruning to Top-{Config.TOP_K_SIMILAR} neighbors..."
            )
            S = SimilarityEngine._prune_top_k(S, k=Config.TOP_K_SIMILAR)

        # 3. Save to cache
        print(f"[SimilarityEngine] Saving similarity matrix to {cache_path}...")
        sp.save_npz(cache_path, S)

        return S

    @staticmethod
    def _prune_top_k(matrix, k):
        """
        Retains only the top-K largest values in each row of the sparse matrix.

        Args:
            matrix (sp.csr_matrix): Input sparse matrix.
            k (int): Number of elements to keep per row.

        Returns:
            sp.csr_matrix: Pruned matrix.
        """
        matrix = matrix.tocsr()
        n_rows = matrix.shape[0]

        # Arrays to build the new sparse matrix
        new_data = []
        new_indices = []
        new_indptr = [0]

        # Iterate over rows
        # While a python loop is generally slow, for ~100k items it is acceptable (~5-10s)
        # compared to the complexity of vectorizing this operation without heavy memory usage.
        for i in range(n_rows):
            start = matrix.indptr[i]
            end = matrix.indptr[i + 1]

            row_data = matrix.data[start:end]
            row_indices = matrix.indices[start:end]

            if len(row_data) <= k:
                # Keep all if fewer than k
                new_data.extend(row_data)
                new_indices.extend(row_indices)
                new_indptr.append(new_indptr[-1] + len(row_data))
            else:
                # Identify top k indices using argpartition (O(n))
                # argpartition returns indices of the top k elements, but not sorted
                top_k_arg = np.argpartition(row_data, -k)[-k:]

                # Extract data and indices
                new_data.extend(row_data[top_k_arg])
                new_indices.extend(row_indices[top_k_arg])
                new_indptr.append(new_indptr[-1] + k)

        # Construct new CSR matrix
        S_pruned = sp.csr_matrix(
            (new_data, new_indices, new_indptr), shape=matrix.shape, dtype=matrix.dtype
        )

        return S_pruned
