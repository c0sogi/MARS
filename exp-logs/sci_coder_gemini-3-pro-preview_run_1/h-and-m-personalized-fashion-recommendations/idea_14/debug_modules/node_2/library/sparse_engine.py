import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from library.config import Config


class SparseEngine:
    """
    Implements the core linear algebra primitives for the Inventory-Gated Dual-Window Cascade.
    Handles construction of sparse interaction matrices, computation of item-item similarity
    graphs, and generation of trend/inventory vectors.
    """

    def build_interaction_matrix(
        self, df: pd.DataFrame, user_count: int, item_count: int
    ) -> sp.csr_matrix:
        """
        Converts a transaction DataFrame into a User x Item sparse CSR matrix.
        Aggregates duplicate (user, item) pairs by summing their occurrences.

        Args:
            df: DataFrame containing 'user_idx' and 'item_idx'.
            user_count: Total number of users (defines matrix rows).
            item_count: Total number of items (defines matrix columns).

        Returns:
            sp.csr_matrix: Shape (user_count, item_count), dtype=float32.
        """
        # Extract indices
        rows = df["user_idx"].values
        cols = df["item_idx"].values

        # Use ones for interaction existence, or counts if multiple purchases
        # Here we use counts to capture intensity, which helps with IDF/Normalization later
        data = np.ones(len(df), dtype=np.float32)

        # Construct matrix
        # Duplicate entries (user buying same item multiple times) are summed by default in csr_matrix constructor
        # if we pass (data, (row, col))
        mat = sp.csr_matrix(
            (data, (rows, cols)), shape=(user_count, item_count), dtype=np.float32
        )

        return mat

    def compute_similarity(
        self,
        interaction_matrix: sp.csr_matrix,
        top_k: int = Config.TOP_K_NEIGHBORS,
        shrinkage: int = Config.SHRINKAGE,
    ) -> sp.csr_matrix:
        """
        Computes the Item-Item Similarity Matrix S = X^T X with IDF weighting and L2 normalization.
        Prunes the result to keep only the top_k neighbors per item.

        Args:
            interaction_matrix: User x Item sparse matrix.
            top_k: Number of neighbors to keep per item.
            shrinkage: Shrinkage parameter for similarity (optional).

        Returns:
            sp.csr_matrix: Item x Item similarity matrix, shape (n_items, n_items).
        """
        print("Computing Similarity Matrix...")
        X = interaction_matrix.copy()
        n_users, n_items = X.shape

        # 1. IDF Weighting (Inverse Document Frequency)
        # Penalize items bought by many users
        # df_i = number of users who bought item i
        # We can get this by converting X to binary and summing columns
        print("  - Applying IDF weighting...")
        X_binary = X.copy()
        X_binary.data[:] = 1.0
        doc_freq = np.array(X_binary.sum(axis=0)).flatten()

        # Avoid division by zero
        idf = np.log1p(n_users) - np.log1p(doc_freq)

        # Apply IDF to columns of X
        # X = X * diag(idf)
        # Efficient way: multiply CSR data by column weights
        # CSR stores data row by row, so this is tricky. CSC is better for column ops.
        X_csc = X.tocsc()
        for j in range(n_items):
            start = X_csc.indptr[j]
            end = X_csc.indptr[j + 1]
            X_csc.data[start:end] *= idf[j]

        X = X_csc.tocsr()

        # 2. L2 Normalization (Row-wise / User-wise)
        # Normalize user vectors so power users don't dominate
        print("  - Applying L2 normalization...")
        X = normalize(X, norm="l2", axis=1)

        # 3. Compute Similarity S = X^T X
        # Result is Item x Item
        print("  - Computing dot product (X^T @ X)...")
        # With 220GB RAM, this operation is generally safe for 100k items if result is sparse-ish.
        # However, the result could theoretically be dense.
        # We rely on the sparsity of user-item interactions.
        S = X.T.dot(X)

        # 4. Pruning (Top-K)
        print(f"  - Pruning to Top-{top_k} neighbors...")
        S = self._prune_matrix(S, top_k)

        # Optional Shrinkage
        if shrinkage > 0:
            S.data = S.data / (1 + shrinkage)

        return S

    def _prune_matrix(self, matrix: sp.csr_matrix, k: int) -> sp.csr_matrix:
        """
        Keeps only the top K largest values in each row of a sparse matrix.
        """
        # If k is larger than number of items, no pruning needed
        if k >= matrix.shape[1]:
            return matrix

        # Prepare new sparse matrix components
        new_data = []
        new_indices = []
        new_indptr = [0]

        # Iterate over rows
        for i in range(matrix.shape[0]):
            start = matrix.indptr[i]
            end = matrix.indptr[i + 1]

            row_data = matrix.data[start:end]
            row_indices = matrix.indices[start:end]

            if len(row_data) <= k:
                # Keep all
                new_data.extend(row_data)
                new_indices.extend(row_indices)
            else:
                # Find top k indices
                # argpartition is efficient O(n)
                top_k_idx = np.argpartition(row_data, -k)[-k:]
                new_data.extend(row_data[top_k_idx])
                new_indices.extend(row_indices[top_k_idx])

            new_indptr.append(len(new_data))

        return sp.csr_matrix(
            (new_data, new_indices, new_indptr), shape=matrix.shape, dtype=matrix.dtype
        )

    def get_similarity_matrix(
        self,
        df_structure: pd.DataFrame,
        user_count: int,
        item_count: int,
        load_cached_data: bool = True,
    ) -> sp.csr_matrix:
        """
        Retrieves the item-item similarity matrix, using cache if available.
        """
        cache_path = Config.CACHE_SIMILARITY_MATRIX

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading Similarity Matrix from cache: {cache_path}")
            return sp.load_npz(cache_path)

        print("Generating Similarity Matrix from scratch...")
        # Build interaction matrix from structure view
        interaction_mat = self.build_interaction_matrix(
            df_structure, user_count, item_count
        )

        # Compute similarity
        sim_matrix = self.compute_similarity(interaction_mat)

        # Save to cache
        print(f"Saving Similarity Matrix to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        sp.save_npz(cache_path, sim_matrix)

        return sim_matrix

    def get_trend_vector(
        self, df_trend: pd.DataFrame, item_count: int, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Computes the global sales count vector for the trend period.
        """
        cache_path = Config.CACHE_GLOBAL_TREND

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading Trend Vector from cache: {cache_path}")
            return np.load(cache_path)

        print("Generating Trend Vector from scratch...")
        # Count sales per item
        counts = df_trend["item_idx"].value_counts()

        # Map to full item vector
        trend_vector = np.zeros(item_count, dtype=np.float32)
        trend_vector[counts.index] = counts.values

        # Save
        print(f"Saving Trend Vector to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, trend_vector)

        return trend_vector

    def get_inventory_mask(
        self, df_inventory: pd.DataFrame, item_count: int, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Creates a binary mask (1/0) for items that appeared in the inventory window.
        """
        cache_path = Config.CACHE_INVENTORY_MASK

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading Inventory Mask from cache: {cache_path}")
            return np.load(cache_path)

        print("Generating Inventory Mask from scratch...")
        active_items = df_inventory["item_idx"].unique()

        mask = np.zeros(item_count, dtype=np.float32)
        mask[active_items] = 1.0

        # Save
        print(f"Saving Inventory Mask to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, mask)

        return mask
