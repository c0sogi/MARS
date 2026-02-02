import numpy as np
import scipy.sparse as sp
import os
import gc
from sklearn.preprocessing import normalize
from library import config
from library import data_processor


class SimilarityOptimizer:
    """
    Constructs and optimizes Item-Item similarity graphs for the recommendation engine.
    Handles behavioral similarity (CF), metadata similarity (Variants), and graph fusion.
    """

    def __init__(self):
        self.working_dir = config.WORKING_DIR
        self.cache_path = config.CACHE_MATRICES_HYBRID
        self.precision = config.PRECISION

    def get_idf_weights(self, interaction_matrix):
        """
        Computes Inverse Document Frequency (IDF) weights for items.
        IDF(i) = log(Total Users / (Number of Users who bought i + 1))

        Args:
            interaction_matrix (sp.csr_matrix): User-Item interaction matrix.

        Returns:
            sp.diags: Diagonal matrix of IDF weights.
        """
        n_users = interaction_matrix.shape[0]
        # Convert to CSC to efficiently count users per item (column)
        csc = interaction_matrix.tocsc()
        # diff(indptr) gives the number of non-zero elements per column
        item_counts = np.diff(csc.indptr)

        # Compute IDF
        idf = np.log(n_users / (item_counts + 1.0))
        idf = idf.astype(self.precision)

        return sp.diags(idf)

    def compute_behavioral_graph(self, train_df, user_map, item_map):
        """
        Computes the Behavioral Similarity Matrix (S_behavior).
        Logic: X^T * X with User Normalization and Item IDF Weighting.

        Args:
            train_df (pd.DataFrame): Training transactions.
            user_map (pd.Series): User ID mapping.
            item_map (pd.Series): Item ID mapping.

        Returns:
            sp.csr_matrix: Item-Item similarity matrix.
        """
        print("Computing Behavioral Graph...")
        n_users = len(user_map)
        n_items = len(item_map)

        # 1. Construct Binary Interaction Matrix (Users x Items)
        X = data_processor.get_interaction_matrix(train_df, n_users, n_items)

        # 2. Row-wise (User) Normalization (L2)
        # Mitigates power-user bias
        print("Normalizing user vectors...")
        X_norm = normalize(X, norm="l2", axis=1)

        # 3. IDF Weighting
        # Penalizes globally popular items
        print("Applying IDF weighting...")
        idf_diag = self.get_idf_weights(X)
        X_weighted = X_norm.dot(idf_diag)

        # 4. Compute Similarity (X^T * X)
        # Result is Items x Items
        print("Computing X^T * X similarity...")
        # Transpose first
        X_weighted_T = X_weighted.T
        S_behavior = X_weighted_T.dot(X_weighted)

        # 5. Remove diagonal (self-similarity)
        S_behavior.setdiag(0)

        # Clean up
        del X, X_norm, X_weighted, X_weighted_T
        gc.collect()

        return S_behavior

    def compute_variant_graph(self, articles_df, item_map):
        """
        Computes the Variant Similarity Matrix (S_variant).
        Wraps the data_processor function.
        """
        return data_processor.get_variant_matrix(articles_df, item_map)

    def prune_graph(self, matrix, k=100):
        """
        Prunes the similarity matrix to keep only the top-K neighbors per item.
        This ensures sparsity and fast inference.

        Args:
            matrix (sp.csr_matrix): Similarity matrix.
            k (int): Number of neighbors to keep.

        Returns:
            sp.csr_matrix: Pruned matrix.
        """
        print(f"Pruning graph to top-{k} neighbors...")
        matrix = matrix.tocsr()

        new_data = []
        new_indices = []
        new_indptr = [0]

        # Iterate over rows
        for i in range(matrix.shape[0]):
            start = matrix.indptr[i]
            end = matrix.indptr[i + 1]

            if end - start <= k:
                # Keep all
                new_data.extend(matrix.data[start:end])
                new_indices.extend(matrix.indices[start:end])
                new_indptr.append(new_indptr[-1] + (end - start))
            else:
                # Get row data
                row_data = matrix.data[start:end]
                row_indices = matrix.indices[start:end]

                # Find top K indices using argpartition (faster than full sort)
                # Note: We want largest values. argpartition puts k-th largest at index -k
                # and larger elements after it.
                if len(row_data) > k:
                    top_k_local_indices = np.argpartition(row_data, -k)[-k:]

                    # Get values
                    top_k_data = row_data[top_k_local_indices]
                    top_k_indices = row_indices[top_k_local_indices]

                    new_data.extend(top_k_data)
                    new_indices.extend(top_k_indices)
                    new_indptr.append(new_indptr[-1] + k)
                else:
                    # Fallback (should be covered by first if, but purely defensive)
                    new_data.extend(row_data)
                    new_indices.extend(row_indices)
                    new_indptr.append(new_indptr[-1] + len(row_data))

        # Reconstruct
        pruned = sp.csr_matrix(
            (new_data, new_indices, new_indptr), shape=matrix.shape, dtype=matrix.dtype
        )

        return pruned

    def fuse_graphs(self, S_behavior, S_variant, alpha):
        """
        Combines behavioral and variant graphs.
        S_hybrid = S_behavior + alpha * S_variant
        """
        print(f"Fusing graphs with lambda={alpha}...")
        # Ensure same shape
        if S_behavior.shape != S_variant.shape:
            raise ValueError(
                f"Shape mismatch: Behavior {S_behavior.shape} vs Variant {S_variant.shape}"
            )

        # Weighted sum
        # Note: S_variant is binary, S_behavior is float
        S_hybrid = S_behavior + (S_variant * alpha)

        return S_hybrid

    def run(self, train_df, articles_df, user_map, item_map, load_cached_data=True):
        """
        Main execution method.
        Checks cache, otherwise computes, fuses, prunes, and saves.

        Args:
            train_df, articles_df: Dataframes.
            user_map, item_map: Mappings.
            load_cached_data (bool): Whether to use cache.

        Returns:
            sp.csr_matrix: The final hybrid similarity matrix.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading hybrid similarity matrix from {self.cache_path}...")
            try:
                return sp.load_npz(self.cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print("Computing similarity graphs from scratch...")

        # 2. Compute Behavioral Graph
        S_behavior = self.compute_behavioral_graph(train_df, user_map, item_map)

        # 3. Compute Variant Graph
        S_variant = self.compute_variant_graph(articles_df, item_map)

        # 4. Fuse Graphs
        # Use lambda from config
        S_hybrid = self.fuse_graphs(S_behavior, S_variant, config.HYBRID_LAMBDA)

        # 5. Prune
        # Keep top 100 neighbors to manage memory/speed for inference
        S_hybrid = self.prune_graph(S_hybrid, k=100)

        # 6. Save Cache
        print(f"Saving hybrid matrix to {self.cache_path}...")
        sp.save_npz(self.cache_path, S_hybrid)

        return S_hybrid
