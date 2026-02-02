import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from library import config


def get_mappings(df, load_cached_data=True):
    """
    Generates or loads user and item mappings to convert string IDs to integer indices.

    Args:
        df (pd.DataFrame): DataFrame containing 'customer_id' and 'article_id'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (user_map, item_map) dictionaries.
    """
    user_map_path = config.CACHE_USER_MAP
    item_map_path = config.CACHE_ITEM_MAP

    # 1. Try Loading from Cache
    if (
        load_cached_data
        and os.path.exists(user_map_path)
        and os.path.exists(item_map_path)
    ):
        print("Loading mappings from cache...")
        user_df = pd.read_parquet(user_map_path)
        item_df = pd.read_parquet(item_map_path)

        user_map = dict(zip(user_df["customer_id"], user_df["user_idx"]))
        item_map = dict(zip(item_df["article_id"], item_df["item_idx"]))
        return user_map, item_map

    # 2. Generate from Scratch
    print("Generating new mappings...")
    unique_users = df["customer_id"].unique()
    unique_items = df["article_id"].unique()

    user_map = {uid: i for i, uid in enumerate(unique_users)}
    item_map = {iid: i for i, iid in enumerate(unique_items)}

    # 3. Save to Cache
    print("Saving mappings to cache...")
    user_df = pd.DataFrame(
        {"customer_id": list(user_map.keys()), "user_idx": list(user_map.values())}
    )
    item_df = pd.DataFrame(
        {"article_id": list(item_map.keys()), "item_idx": list(item_map.values())}
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(user_map_path), exist_ok=True)

    user_df.to_parquet(user_map_path, index=False)
    item_df.to_parquet(item_map_path, index=False)

    return user_map, item_map


def build_decayed_interaction_matrix(df, user_map, item_map, load_cached_data=True):
    """
    Constructs the time-decayed, IDF-weighted interaction matrix X.

    Formula: X_ui = exp(-lambda * days_elapsed) * IDF_i
    Rows are L2 normalized.

    Args:
        df (pd.DataFrame): Transaction data with 'days_elapsed'.
        user_map (dict): Mapping from customer_id to index.
        item_map (dict): Mapping from article_id to index.
        load_cached_data (bool): Whether to load from cache.

    Returns:
        sp.csr_matrix: The normalized interaction matrix (Users x Items).
    """
    cache_path = config.CACHE_INTERACTION_MATRIX

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading interaction matrix from {cache_path}...")
        return sp.load_npz(cache_path)

    print("Building decayed interaction matrix...")

    # Create a working copy with indices
    # We construct mapping DataFrames to merge efficiently
    u_df = pd.DataFrame(list(user_map.items()), columns=["customer_id", "user_idx"])
    i_df = pd.DataFrame(list(item_map.items()), columns=["article_id", "item_idx"])

    # Filter/Map the input dataframe
    # We only keep transactions where user and item are in our maps
    tmp = (
        df[["customer_id", "article_id", "days_elapsed"]]
        .merge(u_df, on="customer_id")
        .merge(i_df, on="article_id")
    )

    # 1. Compute Temporal Weights
    # weight = exp(-lambda * days)
    decay_rate = config.DECAY_RATE
    tmp["weight"] = np.exp(-decay_rate * tmp["days_elapsed"]).astype(np.float32)

    # 2. Compute IDF (Inverse Document Frequency)
    # IDF_i = log(Total_Users / (Users_Bought_i + 1))
    print("Computing IDF...")
    item_user_counts = tmp.groupby("item_idx")["user_idx"].nunique()
    n_users = len(user_map)
    n_items = len(item_map)

    # Create IDF array aligned with item_idx
    idf_values = np.zeros(n_items, dtype=np.float32)
    counts = item_user_counts.values
    indices = item_user_counts.index.values

    # Compute IDF with smoothing
    idf_scores = np.log(n_users / (counts + 1)).astype(np.float32)
    idf_values[indices] = idf_scores

    # 3. Apply IDF to weights
    # Map IDF back to transactions
    tmp["idf"] = idf_values[tmp["item_idx"].values]
    tmp["final_weight"] = tmp["weight"] * tmp["idf"]

    # 4. Construct Sparse Matrix
    print("Constructing sparse matrix...")
    row_ind = tmp["user_idx"].values
    col_ind = tmp["item_idx"].values
    data = tmp["final_weight"].values

    # Sum duplicates (multiple purchases add up)
    interaction_matrix = sp.coo_matrix(
        (data, (row_ind, col_ind)), shape=(n_users, n_items), dtype=np.float32
    ).tocsr()

    # 5. Row-wise L2 Normalization
    # This ensures heavy buyers don't dominate the similarity calculation
    print("Normalizing rows...")
    interaction_matrix = normalize(interaction_matrix, norm="l2", axis=1)

    # 6. Save to Cache
    print(f"Saving interaction matrix to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    sp.save_npz(cache_path, interaction_matrix)

    return interaction_matrix


def compute_similarity_matrix(interaction_matrix, load_cached_data=True):
    """
    Computes Item-Item similarity S = X^T X and prunes to top-K neighbors.

    Args:
        interaction_matrix (sp.csr_matrix): User-Item matrix.
        load_cached_data (bool): Whether to load from cache.

    Returns:
        sp.csr_matrix: Pruned Item-Item similarity matrix.
    """
    cache_path = config.CACHE_SIMILARITY_MATRIX

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading similarity matrix from {cache_path}...")
        return sp.load_npz(cache_path)

    print("Computing similarity matrix (X.T @ X)...")
    # S_{ij} = sum_u (X_{ui} * X_{uj})
    # High score means items are bought by same users (weighted by recency/IDF)
    S = interaction_matrix.T.dot(interaction_matrix)

    # Pruning to top-K neighbors
    print(f"Pruning similarity matrix to top {config.MAX_NEIGHBORS} neighbors...")
    S = _prune_csr(S, k=config.MAX_NEIGHBORS)

    # Zero out diagonal (self-similarity)
    # We want to recommend *other* items, not the one already in the query vector
    print("Zeroing diagonal...")
    S.setdiag(0)
    S.eliminate_zeros()

    # Save to Cache
    print(f"Saving similarity matrix to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    sp.save_npz(cache_path, S)

    return S


def _prune_csr(matrix, k):
    """
    Helper function to prune a CSR matrix to keep only the top K values per row.
    """
    matrix = matrix.tocsr()

    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    n_rows = matrix.shape[0]

    new_data = []
    new_indices = []
    new_indptr = [0]

    # Iterate over rows
    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i + 1]
        row_len = end - start

        if row_len <= k:
            # Keep all elements if row is short
            new_data.append(data[start:end])
            new_indices.append(indices[start:end])
            new_indptr.append(new_indptr[-1] + row_len)
        else:
            # Select top K elements
            row_data = data[start:end]
            row_inds = indices[start:end]

            # argpartition finds the indices of the k largest elements
            # Note: argpartition puts the k-th element in position, smaller to left, larger to right
            # We want the largest k, so we take from -k to end
            top_k_local_indices = np.argpartition(row_data, -k)[-k:]

            new_data.append(row_data[top_k_local_indices])
            new_indices.append(row_inds[top_k_local_indices])
            new_indptr.append(new_indptr[-1] + k)

    # Reconstruct CSR matrix
    if len(new_data) > 0:
        new_data = np.concatenate(new_data)
        new_indices = np.concatenate(new_indices)
    else:
        new_data = np.array([], dtype=np.float32)
        new_indices = np.array([], dtype=np.int32)

    return sp.csr_matrix((new_data, new_indices, new_indptr), shape=matrix.shape)
