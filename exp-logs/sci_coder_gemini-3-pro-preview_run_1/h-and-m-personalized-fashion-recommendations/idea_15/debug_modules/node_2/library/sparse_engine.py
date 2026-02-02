import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from library import config


class SparseEngine:
    """
    Handles low-level sparse matrix operations for the ADIPC model.
    Includes matrix construction, weighting, normalization, and similarity computation.
    """

    def __init__(self):
        pass

    def build_user_item_matrix(self, df, weights, num_users, num_items):
        """
        Constructs a sparse user-item matrix (CSR format).

        Args:
            df (pd.DataFrame): Data containing 'user_idx' and 'article_idx'.
            weights (np.ndarray): Array of weights corresponding to rows in df.
            num_users (int): Total number of users (rows).
            num_items (int): Total number of items (columns).

        Returns:
            scipy.sparse.csr_matrix: The weighted user-item matrix (float32).
        """
        # Extract coordinates and data
        rows = df["user_idx"].values
        cols = df["article_idx"].values
        data = weights.astype(config.FLOAT_DTYPE)

        # Construct COO matrix first (efficient for construction)
        # Duplicate entries (user buying same item multiple times) are summed upon conversion to CSR
        matrix_coo = sp.coo_matrix(
            (data, (rows, cols)), shape=(num_users, num_items), dtype=config.FLOAT_DTYPE
        )

        # Convert to CSR for efficient arithmetic operations
        matrix_csr = matrix_coo.tocsr()

        return matrix_csr

    def apply_idf_weighting(self, matrix):
        """
        Applies IDF (Inverse Document Frequency) weighting to the columns (items).
        IDF = log(Total Users / (Document Frequency + 1))

        Args:
            matrix (scipy.sparse.csr_matrix): User-Item matrix.

        Returns:
            scipy.sparse.csr_matrix: IDF-weighted matrix.
        """
        num_users = matrix.shape[0]

        # Convert to CSC to efficiently calculate column support (Document Frequency)
        matrix_csc = matrix.tocsc()

        # diff(indptr) gives the number of non-zeros in each column
        doc_freqs = np.diff(matrix_csc.indptr)

        # Calculate IDF weights
        # We add 1 to DF to avoid division by zero
        # We use natural log
        idf = np.log(num_users / (doc_freqs + 1.0))
        idf = idf.astype(config.FLOAT_DTYPE)

        # Create a diagonal matrix for broadcasting multiplication
        idf_diag = sp.diags(idf)

        # Apply weighting: Matrix @ Diagonal_Matrix
        # This scales each column j by idf[j]
        weighted_matrix = matrix @ idf_diag

        return weighted_matrix.tocsr()

    def normalize_rows(self, matrix):
        """
        Applies L2 normalization to the rows (users) of the matrix.

        Args:
            matrix (scipy.sparse.csr_matrix): Input matrix.

        Returns:
            scipy.sparse.csr_matrix: Row-normalized matrix.
        """
        # sklearn's normalize is efficient for sparse matrices
        # copy=False attempts to perform in-place if possible
        return normalize(matrix, norm="l2", axis=1, copy=False)

    def compute_item_similarity(self, interaction_matrix, top_k=config.MAX_NEIGHBORS):
        """
        Computes the Item-Item similarity matrix (X^T @ X) and prunes it to the top-K neighbors.

        Args:
            interaction_matrix (scipy.sparse.csr_matrix): Normalized User-Item matrix.
            top_k (int): Number of neighbors to keep per item.

        Returns:
            scipy.sparse.csr_matrix: Pruned Item-Item similarity matrix.
        """
        print("Computing raw item-item similarity (X^T @ X)...")
        # Compute cosine similarity (since rows are L2 normalized)
        # Result is (Items x Items)
        # Note: This step can be memory intensive if the resulting matrix is very dense.
        # Given the sparsity of user-item interactions, X^T X is usually manageable in RAM
        # before pruning, especially with 220GB RAM.
        similarity = interaction_matrix.T @ interaction_matrix

        # Zero out the diagonal (we don't want to recommend the item itself based on similarity to itself)
        similarity.setdiag(0)

        # Remove explicit zeros to save space
        similarity.eliminate_zeros()

        print(f"Pruning similarity matrix to top-{top_k} neighbors per item...")
        pruned_similarity = self._prune_matrix(similarity, top_k)

        return pruned_similarity

    def _prune_matrix(self, matrix, top_k):
        """
        Retains only the top_k largest values in each row of a sparse matrix.
        Optimized using numpy operations and pre-allocation.

        Args:
            matrix (scipy.sparse.csr_matrix): Input matrix.
            top_k (int): K.

        Returns:
            scipy.sparse.csr_matrix: Pruned matrix.
        """
        num_rows = matrix.shape[0]

        # Pre-allocate arrays for the new pruned matrix
        # Maximum possible non-zeros is num_rows * top_k
        max_nnz = num_rows * top_k

        new_data = np.zeros(max_nnz, dtype=config.FLOAT_DTYPE)
        new_indices = np.zeros(max_nnz, dtype=np.int32)
        new_indptr = np.zeros(num_rows + 1, dtype=np.int32)

        # Access internal arrays for speed
        indptr = matrix.indptr
        indices = matrix.indices
        data = matrix.data

        current_ptr = 0

        for i in range(num_rows):
            start = indptr[i]
            end = indptr[i + 1]
            length = end - start

            if length <= top_k:
                # If row has fewer than K elements, keep all
                new_data[current_ptr : current_ptr + length] = data[start:end]
                new_indices[current_ptr : current_ptr + length] = indices[start:end]
                current_ptr += length
            else:
                # If row has more than K elements, select top K
                row_data = data[start:end]

                # argpartition finds the indices of the k largest elements
                # It does not strictly sort them, which is fine for CSR
                # We want the last top_k indices from the partitioned array
                top_k_local_indices = np.argpartition(row_data, -top_k)[-top_k:]

                # Map local indices back to global arrays
                # Note: indices[start + local_idx] gives the correct column index
                new_data[current_ptr : current_ptr + top_k] = row_data[
                    top_k_local_indices
                ]
                new_indices[current_ptr : current_ptr + top_k] = indices[
                    start + top_k_local_indices
                ]
                current_ptr += top_k

            new_indptr[i + 1] = current_ptr

        # Trim arrays to actual size
        new_data = new_data[:current_ptr]
        new_indices = new_indices[:current_ptr]

        # Create new CSR matrix
        pruned_matrix = sp.csr_matrix(
            (new_data, new_indices, new_indptr),
            shape=matrix.shape,
            dtype=config.FLOAT_DTYPE,
        )

        return pruned_matrix
