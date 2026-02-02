import numpy as np
import scipy.sparse as sp
import os
import gc
from library.utils import Timer, set_seed


class ItemSimilarityModel:
    """
    Computes and manages the Item-Item similarity graph for the Decay-Weighted Behavioral Cascade.
    Implements efficient block-wise matrix multiplication and pruning to handle high-dimensional data.
    """

    def __init__(self, cache_dir="./working/idea_8"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.similarity_matrix = None

    def compute_similarity(
        self, X, top_k=100, batch_size=1000, load_cached_data=True, save=True
    ):
        """
        Computes the pruned Item-Item similarity matrix S = X^T * X.

        The computation is performed in batches to manage memory usage. For each batch of items,
        we compute their similarity to all other items, prune to the top_k neighbors, and
        store the result. This avoids materializing the full dense similarity matrix.

        Args:
            X (csr_matrix): The normalized, decay-weighted interaction matrix (Users x Items).
            top_k (int): Number of most similar neighbors to retain per item.
            batch_size (int): Number of items (columns) to process in a single block.
            load_cached_data (bool): If True, attempts to load the matrix from disk.
            save (bool): If True, saves the computed matrix to disk.

        Returns:
            scipy.sparse.csr_matrix: The pruned similarity matrix S (Items x Items).
        """
        cache_path = os.path.join(self.cache_dir, f"similarity_matrix_k{top_k}.npz")

        # 1. Attempt to Load from Cache
        if load_cached_data:
            if os.path.exists(cache_path):
                print(
                    f"[ItemSimilarityModel] Loading cached similarity matrix from {cache_path}..."
                )
                with Timer("Load Similarity Matrix"):
                    self.similarity_matrix = sp.load_npz(cache_path)
                return self.similarity_matrix
            else:
                print(
                    f"[ItemSimilarityModel] Cache miss at {cache_path}. Computing from scratch..."
                )
        else:
            print(
                "[ItemSimilarityModel] Force reload requested. Computing from scratch..."
            )

        # 2. Prepare for Computation
        n_users, n_items = X.shape
        print(f"  Input Interaction Matrix Shape: {X.shape}")

        # We need X in CSC format for efficient column slicing (to get item vectors)
        # X is typically CSR coming from the factory.
        with Timer("Convert X to CSC"):
            X_csc = X.tocsc()

        # Lists to accumulate the sparse matrix components
        final_rows = []
        final_cols = []
        final_data = []

        # 3. Block-wise Computation and Pruning
        # We compute rows of the similarity matrix S.
        # Since S = X^T * X, the i-th row of S corresponds to the dot product of
        # the i-th column of X (item i) with all other columns of X.
        # S_row_i = (X[:, i])^T * X

        print(f"  Computing similarity in batches of {batch_size} items...")

        with Timer("Compute & Prune Batches"):
            for start_col in range(0, n_items, batch_size):
                end_col = min(start_col + batch_size, n_items)
                current_batch_size = end_col - start_col

                # Extract a batch of item vectors (Users x BatchSize)
                X_batch = X_csc[:, start_col:end_col]

                # Transpose to (BatchSize x Users) to prepare for dot product
                # Resulting shape of dot product will be (BatchSize x n_items)
                # These are the rows of S corresponding to the items in the batch.
                X_batch_T = X_batch.T.tocsr()

                # Compute raw similarity scores for this batch
                # S_block[i, j] = Similarity between batch_item[i] and global_item[j]
                S_block = X_batch_T.dot(X)

                # Convert to dense array to perform efficient top-k selection
                # A 1000 x 100,000 float32 block is ~400MB, which fits easily in memory.
                S_dense = S_block.toarray()

                # Iterate through each item in the batch to prune
                for i in range(current_batch_size):
                    global_item_idx = start_col + i
                    row_vec = S_dense[i]

                    # Zero out self-similarity to prevent recommending the item itself
                    # (unless we explicitly want to reinforce repurchases here, but
                    # the cascade logic separates Habitual Repurchase into Stratum 1)
                    row_vec[global_item_idx] = 0.0

                    # Identify Top-K neighbors
                    # np.argpartition is O(n) and faster than full sort
                    if n_items > top_k:
                        # Puts top k elements at the end of the array
                        top_indices = np.argpartition(row_vec, -top_k)[-top_k:]
                    else:
                        top_indices = np.arange(n_items)

                    # Retrieve values and filter zeros
                    top_values = row_vec[top_indices]

                    mask = top_values > 1e-6  # Filter out near-zero similarities
                    top_indices = top_indices[mask]
                    top_values = top_values[mask]

                    if len(top_indices) > 0:
                        # Store in COO format lists
                        # We repeat the row index (global_item_idx) for each neighbor
                        final_rows.append(
                            np.full(len(top_indices), global_item_idx, dtype=np.int32)
                        )
                        final_cols.append(top_indices.astype(np.int32))
                        final_data.append(top_values.astype(np.float32))

                # Periodic garbage collection
                if (start_col // batch_size) % 10 == 0:
                    gc.collect()

        # 4. Construct Final Sparse Matrix
        with Timer("Construct Final CSR"):
            if final_rows:
                all_rows = np.concatenate(final_rows)
                all_cols = np.concatenate(final_cols)
                all_data = np.concatenate(final_data)

                self.similarity_matrix = sp.csr_matrix(
                    (all_data, (all_rows, all_cols)),
                    shape=(n_items, n_items),
                    dtype=np.float32,
                )
            else:
                # Handle edge case of no similarities
                self.similarity_matrix = sp.csr_matrix(
                    (n_items, n_items), dtype=np.float32
                )

        print(f"  Final Similarity Matrix Shape: {self.similarity_matrix.shape}")
        print(f"  Non-zero elements: {self.similarity_matrix.nnz}")
        print(f"  Density: {self.similarity_matrix.nnz / (n_items * n_items):.6f}")

        # 5. Save to Cache
        if save:
            print(f"[ItemSimilarityModel] Saving matrix to {cache_path}...")
            with Timer("Save Artifacts"):
                sp.save_npz(cache_path, self.similarity_matrix)

        return self.similarity_matrix

    def get_similarity(self):
        """Returns the currently loaded similarity matrix."""
        return self.similarity_matrix
