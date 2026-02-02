import os
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from datetime import timedelta

from library import config, data_loader, visual_encoder


def _get_global_article_map(articles_df):
    """
    Creates a mapping from article_id to a dense integer index (0 to N-1).
    Returns:
        id_to_idx (dict): Mapping from article_id to index.
        idx_to_id (np.array): Array where index i contains the article_id.
    """
    # Ensure sorted order for consistency
    unique_ids = np.sort(articles_df["article_id"].unique())
    id_to_idx = {aid: i for i, aid in enumerate(unique_ids)}
    return id_to_idx, unique_ids


def _get_global_customer_map(customers_df):
    """
    Creates a mapping from customer_id to a dense integer index.
    """
    unique_ids = customers_df["customer_id"].unique()
    # We don't necessarily need sorted order for customers, but it helps stability
    # However, customers are strings (hashes), sorting is expensive but fine for 1.3M
    unique_ids = np.sort(unique_ids)
    id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
    return id_to_idx, unique_ids


def build_sequential_graph(load_cached_data=True):
    """
    Constructs the Time-Aware Sequential Transition Matrix (T_seq).

    Logic:
    1. Filter transactions to the last RECENCY_WEEKS.
    2. Identify consecutive purchases (t, t+1) by the same user.
    3. Count transitions to form edge weights.
    4. Save as sparse CSR matrix.
    """
    # 1. Check Cache
    if load_cached_data and config.TRANSITION_MATRIX_PATH.exists():
        print(
            f"Loading cached sequential graph from {config.TRANSITION_MATRIX_PATH}..."
        )
        return sp.load_npz(config.TRANSITION_MATRIX_PATH)

    print("Building sequential graph from scratch...")

    # 2. Load Data
    # We need the global article map to ensure matrix dimensions match articles.csv
    articles_df = data_loader.load_articles(load_cached_data=True)
    art_id_to_idx, _ = _get_global_article_map(articles_df)
    n_articles = len(art_id_to_idx)

    # Load transactions
    df = data_loader.load_transactions("train", load_cached_data=True)

    # 3. Filter by Recency
    max_date = df["t_dat"].max()
    cutoff_date = max_date - timedelta(weeks=config.RECENCY_WEEKS)
    print(f"Filtering transactions after {cutoff_date}...")
    df = df[df["t_dat"] > cutoff_date].copy()

    # 4. Map Article IDs to Dense Indices
    # Filter out transactions with unknown articles (should be 0 if articles.csv is master)
    df = df[df["article_id"].isin(art_id_to_idx)].copy()
    df["article_idx"] = df["article_id"].map(art_id_to_idx).astype(np.int32)

    # 5. Identify Transitions
    print("Sorting and identifying transitions...")
    # Sort by customer and date to ensure sequential order
    df.sort_values(["customer_id", "t_dat"], inplace=True)

    # Convert to numpy for fast array manipulation
    customer_ids = df["customer_id"].values
    article_indices = df["article_idx"].values

    # Identify boundaries where customer changes
    # mask[i] is True if row i and row i+1 belong to same customer
    # We compare customer_ids[:-1] with customer_ids[1:]
    same_customer_mask = customer_ids[:-1] == customer_ids[1:]

    # Source nodes (t) and Target nodes (t+1)
    u = article_indices[:-1][same_customer_mask]
    v = article_indices[1:][same_customer_mask]

    # 6. Build Sparse Matrix
    print(f"Constructing sparse matrix with {len(u)} transitions...")
    # We use raw counts (weight=1 for each transition occurrence)
    data = np.ones(len(u), dtype=np.float32)

    # Shape is (N_articles, N_articles)
    adj_matrix = sp.coo_matrix((data, (u, v)), shape=(n_articles, n_articles))

    # Sum duplicates to get raw counts
    adj_matrix = adj_matrix.tocsr()
    # adj_matrix.sum_duplicates() is implicit in tocsr conversion usually, but good to be sure
    # However, coo to csr sums duplicates by default.

    print(f"Sequential Graph Shape: {adj_matrix.shape}, NNZ: {adj_matrix.nnz}")

    # 7. Save
    print(f"Saving sequential graph to {config.TRANSITION_MATRIX_PATH}...")
    sp.save_npz(config.TRANSITION_MATRIX_PATH, adj_matrix)

    # Cleanup
    del df, customer_ids, article_indices, u, v, data
    gc.collect()

    return adj_matrix


def build_visual_graph(load_cached_data=True):
    """
    Constructs the Visual K-Nearest Neighbors Graph (T_vis).

    Logic:
    1. Get image embeddings.
    2. Map embedding article IDs to global indices.
    3. Compute Top-K Cosine Similarity neighbors.
    4. Save as sparse CSR matrix.
    """
    # 1. Check Cache
    if load_cached_data and config.VISUAL_GRAPH_PATH.exists():
        print(f"Loading cached visual graph from {config.VISUAL_GRAPH_PATH}...")
        return sp.load_npz(config.VISUAL_GRAPH_PATH)

    print("Building visual graph from scratch...")

    # 2. Load Embeddings and Global Map
    # embeddings: (N_images, 512), emb_article_ids: (N_images,)
    embeddings, emb_article_ids = visual_encoder.generate_embeddings(
        load_cached_data=True
    )

    articles_df = data_loader.load_articles(load_cached_data=True)
    global_id_to_idx, _ = _get_global_article_map(articles_df)
    n_articles = len(global_id_to_idx)

    # 3. Map Embeddings to Global Space
    print("Mapping embeddings to global index space...")
    valid_indices = []
    valid_emb_rows = []

    for i, aid in enumerate(emb_article_ids):
        if aid in global_id_to_idx:
            valid_indices.append(global_id_to_idx[aid])
            valid_emb_rows.append(i)

    valid_indices = np.array(valid_indices)
    valid_embeddings = embeddings[valid_emb_rows]

    print(f"Valid embeddings mapped: {len(valid_embeddings)} out of {len(embeddings)}")

    # 4. Compute KNN
    print(f"Computing {config.VISUAL_KNN_K}-Nearest Neighbors...")
    # Metric: cosine. Note: embeddings are already L2 normalized by visual_encoder.
    # So euclidean distance is equivalent to cosine similarity ranking.
    # Cosine Sim = 1 - (Euclidean^2) / 2 for normalized vectors.
    # We use 'cosine' metric directly for clarity, though 'euclidean' on normalized might be faster in some impls.
    knn = NearestNeighbors(n_neighbors=config.VISUAL_KNN_K, metric="cosine", n_jobs=-1)
    knn.fit(valid_embeddings)

    # Find neighbors for the valid items
    distances, neighbor_indices_local = knn.kneighbors(valid_embeddings)

    # 5. Construct Sparse Matrix
    print("Constructing sparse visual matrix...")
    # Convert distances to similarity scores. Cosine dist is 1 - sim.
    # Sim = 1 - dist.
    similarities = 1.0 - distances

    # Threshold to remove weak links (optional, but good for sparsity)
    # Keeping all K is fine as K is small (20).

    # We need to map the neighbor indices (which are 0..len(valid_embeddings)-1)
    # back to the global indices.
    # neighbor_indices_local contains indices into `valid_indices`.

    # Flatten for COO construction
    row_indices_local = np.repeat(np.arange(len(valid_indices)), config.VISUAL_KNN_K)
    col_indices_local = neighbor_indices_local.flatten()
    data = similarities.flatten()

    # Map local indices to global indices
    # valid_indices[i] gives the global index for local index i
    row_indices_global = valid_indices[row_indices_local]
    col_indices_global = valid_indices[col_indices_local]

    # Construct matrix (N_articles, N_articles)
    vis_matrix = sp.coo_matrix(
        (data, (row_indices_global, col_indices_global)), shape=(n_articles, n_articles)
    ).tocsr()

    print(f"Visual Graph Shape: {vis_matrix.shape}, NNZ: {vis_matrix.nnz}")

    # 6. Save
    print(f"Saving visual graph to {config.VISUAL_GRAPH_PATH}...")
    sp.save_npz(config.VISUAL_GRAPH_PATH, vis_matrix)

    # Cleanup
    del (
        embeddings,
        valid_embeddings,
        knn,
        distances,
        row_indices_global,
        col_indices_global,
    )
    gc.collect()

    return vis_matrix


def build_user_history(load_cached_data=True):
    """
    Constructs a sparse User-Item interaction matrix (U_history).
    Used for the repurchase signal and fast history lookup.

    Rows: Customers (Global Index)
    Cols: Articles (Global Index)
    Values: Purchase counts (or 1 for binary)
    """
    # 1. Check Cache
    if load_cached_data and config.USER_HISTORY_PATH.exists():
        print(f"Loading cached user history from {config.USER_HISTORY_PATH}...")
        return sp.load_npz(config.USER_HISTORY_PATH)

    print("Building user history matrix from scratch...")

    # 2. Load Data
    articles_df = data_loader.load_articles(load_cached_data=True)
    customers_df = data_loader.load_customers(load_cached_data=True)
    transactions_df = data_loader.load_transactions("train", load_cached_data=True)

    # 3. Map IDs
    art_id_to_idx, _ = _get_global_article_map(articles_df)
    cust_id_to_idx, _ = _get_global_customer_map(customers_df)

    n_articles = len(art_id_to_idx)
    n_customers = len(cust_id_to_idx)

    # Filter valid
    transactions_df = transactions_df[
        transactions_df["article_id"].isin(art_id_to_idx)
        & transactions_df["customer_id"].isin(cust_id_to_idx)
    ].copy()

    # Map
    row_idx = transactions_df["customer_id"].map(cust_id_to_idx).astype(np.int32).values
    col_idx = transactions_df["article_id"].map(art_id_to_idx).astype(np.int32).values
    data = np.ones(len(row_idx), dtype=np.float32)

    # 4. Build Matrix
    print(f"Constructing User-Item matrix ({n_customers} x {n_articles})...")
    user_item_matrix = sp.coo_matrix(
        (data, (row_idx, col_idx)), shape=(n_customers, n_articles)
    ).tocsr()

    # Sum duplicates (count purchases)
    # Note: If we just want binary history, we could use .astype(bool).astype(float)
    # But repurchase count is a useful signal.

    print(f"User History Shape: {user_item_matrix.shape}, NNZ: {user_item_matrix.nnz}")

    # 5. Save
    print(f"Saving user history to {config.USER_HISTORY_PATH}...")
    sp.save_npz(config.USER_HISTORY_PATH, user_item_matrix)

    # Cleanup
    del transactions_df, row_idx, col_idx, data
    gc.collect()

    return user_item_matrix
