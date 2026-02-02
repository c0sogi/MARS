import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from library.config import Config
from library.data_utils import IndexMapper


def compute_behavioral_similarity_matrix(
    df, mapper: IndexMapper, load_cached_data: bool = True
):
    """
    Computes the Item-Item Behavioral Similarity Matrix (S_behavior).

    Logic:
    1. Construct User-Item interaction matrix (R).
    2. Apply IDF Weighting to columns (Items) to down-weight universally popular items.
    3. Apply L2 Normalization to rows (Users) to mitigate power-user bias.
    4. Compute S = R.T @ R (Cosine similarity on weighted vectors).
    5. Prune to Top-K neighbors per item.

    Args:
        df (pd.DataFrame): Transaction DataFrame.
        mapper (IndexMapper): Fitted mapper instance.
        load_cached_data (bool): Whether to use cached .npz file.

    Returns:
        scipy.sparse.csr_matrix: Sparse similarity matrix of shape (n_items, n_items).
    """
    cache_path = Config.CACHE_SIM_BEHAVIOR
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading behavioral similarity matrix from cache: {cache_path}")
        try:
            matrix = sp.load_npz(cache_path)
            expected_shape = (mapper.get_num_items(), mapper.get_num_items())
            if matrix.shape == expected_shape:
                return matrix
            else:
                print(
                    f"Cached matrix shape {matrix.shape} mismatch (expected {expected_shape}). Recomputing..."
                )
        except Exception as e:
            print(f"Failed to load behavioral cache: {e}. Recomputing...")

    if df is None:
        raise ValueError(
            "DataFrame 'df' is required when cache is missing or load_cached_data=False."
        )

    print("Computing behavioral similarity matrix...")

    # Map indices
    user_indices = mapper.map_users(df["customer_id"])
    item_indices = mapper.map_items(df["article_id"])

    # Filter valid
    valid_mask = (user_indices >= 0) & (item_indices >= 0)
    user_indices = user_indices[valid_mask]
    item_indices = item_indices[valid_mask]

    n_users = mapper.get_num_users()
    n_items = mapper.get_num_items()

    # 1. Construct Binary User-Item Matrix
    # We use ones for existence of purchase in the window
    print("Constructing interaction matrix...")
    # Remove duplicates (user bought same item multiple times in window) for binary occurrence
    # We want co-occurrence of *types* of items, not volume here (volume is in Repurchase View)
    unique_pairs = pd.DataFrame(
        {"u": user_indices, "i": item_indices}
    ).drop_duplicates()

    data = np.ones(len(unique_pairs), dtype=np.float32)
    R = sp.csr_matrix(
        (data, (unique_pairs["u"].values, unique_pairs["i"].values)),
        shape=(n_users, n_items),
    )

    # 2. IDF Weighting (Column-wise)
    print("Applying IDF weighting...")
    # Document frequency = number of users who bought the item
    # axis=0 counts non-zeros per column (item)
    item_counts = np.array(R.getnnz(axis=0)).astype(np.float32)

    # Avoid division by zero
    item_counts[item_counts == 0] = 1.0

    # IDF = log(Total Users / Item Frequency)
    idf = np.log(n_users / (1.0 + item_counts))

    # Apply IDF to columns.
    # Efficient way: Multiply R by diagonal matrix of IDFs
    # R_idf = R @ diag(idf)
    R_idf = R.multiply(idf[np.newaxis, :])  # Broadcasting multiply

    # 3. User Normalization (Row-wise)
    print("Applying User L2 normalization...")
    # Normalize rows to unit L2 norm
    R_norm = normalize(R_idf, norm="l2", axis=1)

    # 4. Compute Similarity S = R.T @ R
    print("Computing R.T @ R...")
    # Result is (n_items, n_items)
    # Since we normalized rows, this effectively computes cosine similarity
    # between items based on the weighted user vectors.
    S = R_norm.T @ R_norm

    # 5. Prune to Top-K
    print(f"Pruning to Top-{Config.TOP_K_SIMILAR} neighbors...")
    S = _prune_similarity_matrix(S, k=Config.TOP_K_SIMILAR)

    # Save Cache
    print(f"Saving behavioral similarity to {cache_path}...")
    sp.save_npz(cache_path, S)

    print(f"Behavioral Similarity Matrix ready. Shape: {S.shape}, NNZ: {S.nnz}")
    return S


def _prune_similarity_matrix(matrix, k=20):
    """
    Retains only the top K values per row in a sparse matrix.
    Also removes diagonal (self-similarity) if present.
    """
    matrix = matrix.tocsr()
    n_rows = matrix.shape[0]

    # Prepare lists for new sparse matrix construction
    new_data = []
    new_indices = []
    new_indptr = [0]

    for i in range(n_rows):
        # Get row slice
        start = matrix.indptr[i]
        end = matrix.indptr[i + 1]

        row_data = matrix.data[start:end]
        row_inds = matrix.indices[start:end]

        if len(row_data) == 0:
            new_indptr.append(new_indptr[-1])
            continue

        # Filter out self-similarity (diagonal)
        mask = row_inds != i
        row_data = row_data[mask]
        row_inds = row_inds[mask]

        if len(row_data) > k:
            # Get indices of top K values
            # argpartition puts top K elements at the end
            top_k_idx = np.argpartition(row_data, -k)[-k:]

            # Select them
            best_data = row_data[top_k_idx]
            best_inds = row_inds[top_k_idx]

            # Sort by value descending (optional but nice for debugging)
            sort_order = np.argsort(-best_data)
            new_data.append(best_data[sort_order])
            new_indices.append(best_inds[sort_order])
            new_indptr.append(new_indptr[-1] + k)
        else:
            # Keep all if less than K
            # Sort descending
            sort_order = np.argsort(-row_data)
            new_data.append(row_data[sort_order])
            new_indices.append(row_inds[sort_order])
            new_indptr.append(new_indptr[-1] + len(row_data))

    # Concatenate
    if new_data:
        new_data = np.concatenate(new_data)
        new_indices = np.concatenate(new_indices)
    else:
        new_data = np.array([])
        new_indices = np.array([])

    return sp.csr_matrix((new_data, new_indices, new_indptr), shape=matrix.shape)


def calculate_global_trend(df, mapper: IndexMapper, load_cached_data: bool = True):
    """
    Computes a global popularity vector (V_trend) based on time-decayed sales.

    Args:
        df (pd.DataFrame): Transaction DataFrame.
        mapper (IndexMapper): Fitted mapper instance.
        load_cached_data (bool): Whether to use cached .parquet file.

    Returns:
        np.ndarray: Dense array of shape (n_items,) with normalized scores.
    """
    cache_path = Config.CACHE_GLOBAL_TRENDS
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading global trends from cache: {cache_path}")
        try:
            trend_df = pd.read_parquet(cache_path)
            # Ensure it covers all items
            n_items = mapper.get_num_items()
            trend_vector = np.zeros(n_items, dtype=np.float32)

            # Map cached values to vector
            indices = trend_df["item_idx"].values
            scores = trend_df["score"].values

            # Filter indices that might be out of bounds (if mapper changed)
            mask = indices < n_items
            trend_vector[indices[mask]] = scores[mask]

            print(f"Loaded trend vector shape: {trend_vector.shape}")
            return trend_vector
        except Exception as e:
            print(f"Failed to load trend cache: {e}. Recomputing...")

    if df is None:
        raise ValueError(
            "DataFrame 'df' is required when cache is missing or load_cached_data=False."
        )

    print("Computing global trend vector...")

    # Calculate Time Decay Weights
    max_date = df["t_dat"].max()
    days_diff = (max_date - df["t_dat"]).dt.days

    # Weight formula: 1 / (days + 1)
    # This emphasizes recent trends strongly
    weights = 1.0 / (days_diff + 1.0)

    # Map items
    item_indices = mapper.map_items(df["article_id"])

    # Aggregate
    temp_df = pd.DataFrame({"item_idx": item_indices, "weight": weights})

    # Filter valid items
    temp_df = temp_df[temp_df["item_idx"] >= 0]

    # Sum weights per item
    trend_series = temp_df.groupby("item_idx")["weight"].sum()

    # Construct dense vector
    n_items = mapper.get_num_items()
    trend_vector = np.zeros(n_items, dtype=np.float32)

    indices = trend_series.index.values
    values = trend_series.values

    trend_vector[indices] = values

    # Normalize to [0, 1] range for stability in the ensemble
    max_val = trend_vector.max()
    if max_val > 0:
        trend_vector = trend_vector / max_val

    # Save Cache
    print(f"Saving global trends to {cache_path}...")
    # Save as sparse-like dataframe to save space (many items have 0 sales in window)
    non_zero_mask = trend_vector > 0
    save_df = pd.DataFrame(
        {"item_idx": np.where(non_zero_mask)[0], "score": trend_vector[non_zero_mask]}
    )
    save_df.to_parquet(cache_path, index=False)

    print(f"Global Trend Vector ready. Shape: {trend_vector.shape}")
    return trend_vector
