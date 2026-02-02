import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import os
import gc
from library import config


def build_decayed_interaction_matrix(
    transactions_df, user_to_idx, item_to_idx, load_cached_data=True
):
    """
    Constructs the Time-Decayed User-Item Interaction Matrix.

    Logic:
    1. Calculate days elapsed for each transaction relative to the max date.
    2. Compute weight = exp(-lambda * days_elapsed).
    3. Construct Sparse Matrix (User x Item).
    4. Apply IDF Weighting to Items.
    5. Apply L2 Normalization to Users.

    Args:
        transactions_df (pd.DataFrame): Transaction history.
        user_to_idx (dict): Mapping from customer_id to index.
        item_to_idx (dict): Mapping from article_id to index.
        load_cached_data (bool): Whether to use disk caching.

    Returns:
        scipy.sparse.csr_matrix: The normalized, weighted interaction matrix.
    """
    cache_path = os.path.join(config.CACHE_DIR, "interaction_matrix.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached interaction matrix from {cache_path}...")
        return sp.load_npz(cache_path)

    print("Building decayed interaction matrix...")

    # Ensure working directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # 1. Temporal Decay Calculation
    # We assume transactions_df has 't_dat' converted to datetime
    max_date = transactions_df["t_dat"].max()

    # Calculate days elapsed (vectorized)
    # (max_date - t_dat) returns Timedelta, .dt.days gets integers
    days_elapsed = (max_date - transactions_df["t_dat"]).dt.days.values.astype(
        config.FLOAT_DTYPE
    )

    # Apply Exponential Decay
    # weight = e^(-lambda * t)
    weights = np.exp(-config.DECAY_LAMBDA * days_elapsed).astype(config.FLOAT_DTYPE)

    # 2. Map IDs to Indices
    # Filter transactions to only those with known users/items (safety check)
    # In this pipeline, mappings are usually generated from the same df, so coverage is 100%
    # But we map efficiently using map/replace or by assuming order if pre-sorted.
    # Given the size, using map on the series is standard.

    print("Mapping IDs to indices...")
    # It's faster to map if we ensure the input is clean.
    # We assume user_to_idx covers all users in transactions_df.

    # Optimization: Use pandas map, fillna with -1 to catch errors if any
    u_indices = (
        transactions_df["customer_id"].map(user_to_idx).fillna(-1).astype(np.int32)
    )
    i_indices = (
        transactions_df["article_id"].map(item_to_idx).fillna(-1).astype(np.int32)
    )

    # Filter out invalid indices (if any)
    valid_mask = (u_indices >= 0) & (i_indices >= 0)
    u_indices = u_indices[valid_mask]
    i_indices = i_indices[valid_mask]
    weights = weights[valid_mask]

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    # 3. Construct COO Matrix
    # Duplicate (user, item) entries will be summed automatically when converting to CSR/CSC
    print(f"Constructing sparse matrix ({n_users} x {n_items})...")
    interaction_matrix = sp.coo_matrix(
        (weights, (u_indices, i_indices)),
        shape=(n_users, n_items),
        dtype=config.FLOAT_DTYPE,
    ).tocsr()

    # 4. Apply IDF Weighting (Item Inverse Document Frequency)
    # IDF_i = log(Total Users / Number of Users who bought item i)
    # Note: We use the number of non-zero entries per column.
    print("Applying IDF weighting...")

    # Get document frequency (number of users who interacted with each item)
    # Since interaction_matrix sums weights, we need binary occurrence for IDF.
    # We can get this from the CSC structure's indptr diffs or just counting non-zeros.
    # However, since we already have the weighted matrix, simply counting non-zeros per column is valid
    # provided the weights are non-zero (which exp decay is).

    # Convert to CSC for efficient column operations
    interaction_csc = interaction_matrix.tocsc()

    # Count non-zeros per column
    item_frequencies = np.diff(interaction_csc.indptr)

    # Avoid division by zero
    item_frequencies = np.maximum(item_frequencies, 1)

    # Compute IDF
    # standard formula: log(N / df)
    idf = np.log(n_users / item_frequencies).astype(config.FLOAT_DTYPE)

    # Apply IDF: Multiply each column j by idf[j]
    # Efficient way: Matrix * Diagonal Matrix
    idf_diag = sp.diags(
        idf, offsets=0, shape=(n_items, n_items), format="csr", dtype=config.FLOAT_DTYPE
    )
    interaction_matrix = interaction_matrix.dot(idf_diag)

    # 5. L2 Normalization (Rows/Users)
    # This ensures that users with many purchases don't dominate the similarity dot product
    print("Applying L2 normalization to users...")
    interaction_matrix = normalize(interaction_matrix, norm="l2", axis=1)

    # Save to cache
    print(f"Saving interaction matrix to {cache_path}...")
    sp.save_npz(cache_path, interaction_matrix)

    return interaction_matrix


def compute_similarity_matrix(
    interaction_matrix, top_k=config.SIMILARITY_TOP_K, load_cached_data=True
):
    """
    Computes the Item-Item Similarity Matrix (S = X^T * X).

    Logic:
    1. Transpose Interaction Matrix to get Item-User matrix.
    2. Perform Matrix Multiplication in chunks to avoid OOM.
    3. For each chunk, keep only Top-K values per item.
    4. Assemble final Sparse Matrix.

    Args:
        interaction_matrix (scipy.sparse.csr_matrix): User-Item matrix.
        top_k (int): Number of neighbors to keep per item.
        load_cached_data (bool): Whether to use disk caching.

    Returns:
        scipy.sparse.csr_matrix: Item-Item similarity matrix.
    """
    cache_path = os.path.join(config.CACHE_DIR, "similarity_matrix.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached similarity matrix from {cache_path}...")
        return sp.load_npz(cache_path)

    print(f"Computing similarity matrix with Top-K={top_k} pruning...")

    # Ensure working directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # X is (Users x Items)
    # We want S = X^T * X (Items x Items)
    # Since X is row-normalized, this computes Cosine Similarity between items
    # based on the user vectors who bought them.

    X = interaction_matrix
    X_t = X.T.tocsr()  # (Items x Users)

    n_items = X.shape[1]

    # Chunked Computation
    # We compute columns of S in batches.
    # S[:, j] = X^T * X[:, j]
    # To do this efficiently, we iterate over chunks of columns of X.

    chunk_size = (
        1000  # Adjust based on memory. 1000 items * 100k items result is manageable.
    )

    data = []
    rows = []
    cols = []

    print(f"Total Items: {n_items}. Processing in chunks of {chunk_size}...")

    # Ensure X is CSC for efficient column slicing
    X_csc = X.tocsc()

    for start_idx in range(0, n_items, chunk_size):
        end_idx = min(start_idx + chunk_size, n_items)

        if start_idx % 5000 == 0:
            print(f"Processing items {start_idx} to {end_idx}...")

        # Get chunk of X (Users x Chunk_Items)
        X_chunk = X_csc[:, start_idx:end_idx]

        # Compute Similarity Chunk: (Items x Users) * (Users x Chunk_Items) -> (Items x Chunk_Items)
        # Result is dense-ish, but we densify to use argpartition efficiently
        sim_chunk = X_t.dot(X_chunk)

        # We need to prune this chunk to Top-K per column
        # Since sim_chunk is (n_items, chunk_width), we iterate over the chunk columns

        # Convert to dense for fast top-k selection if memory allows
        # n_items ~100k, chunk ~1k -> 100M floats -> 400MB. Safe.
        sim_dense = sim_chunk.toarray()

        # For each column in the chunk (which corresponds to an item j)
        for i in range(sim_dense.shape[1]):
            col_idx = start_idx + i  # Actual item index
            col_vec = sim_dense[:, i]

            # Self-similarity is usually 1.0 (or close to it).
            # We usually want to keep it or zero it?
            # In item-item CF, we usually keep it or ignore it.
            # Let's keep it, but ensure we pick the best neighbors.

            # Argpartition to find top K indices
            # We want top_k largest.
            if len(col_vec) > top_k:
                # argpartition puts the k-th element in sorted position,
                # all smaller before, all larger after.
                # We want larger.
                ind = np.argpartition(col_vec, -top_k)[-top_k:]

                # Get values
                vals = col_vec[ind]

                # Store
                # We are building a matrix where S[i, j] is similarity between item i and item j.
                # Here col_vec represents S[:, col_idx].
                # So rows are 'ind', col is 'col_idx'.
                rows.extend(ind)
                cols.extend([col_idx] * len(ind))
                data.extend(vals)
            else:
                # Keep all non-zeros
                ind = np.nonzero(col_vec)[0]
                vals = col_vec[ind]
                rows.extend(ind)
                cols.extend([col_idx] * len(ind))
                data.extend(vals)

        # Garbage collection to keep memory clean
        del sim_chunk, sim_dense
        gc.collect()

    print("Constructing final sparse similarity matrix...")
    similarity_matrix = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_items, n_items), dtype=config.FLOAT_DTYPE
    )

    print(f"Saving similarity matrix to {cache_path}...")
    sp.save_npz(cache_path, similarity_matrix)

    return similarity_matrix
