import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from library.config import Config


class SimilarityEngine:
    """
    Constructs sparse similarity matrices for Collaborative Filtering.
    Implements caching and efficient sparse matrix operations.
    """

    def __init__(self):
        self.config = Config
        self.cache_dir = self.config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def build_user_item_matrix(self, train_df, indexer, load_cached_data=True):
        """
        Constructs the time-weighted User-Item history matrix (U).

        Args:
            train_df (pd.DataFrame): Transaction data with 'customer_id', 'article_id', 'days_elapsed'.
            indexer (Indexer): Object containing user and item mappings.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            scipy.sparse.csr_matrix: The user-item matrix U.
        """
        cache_path = os.path.join(self.cache_dir, "U_matrix.npz")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached User-Item matrix from {cache_path}...")
            return sp.load_npz(cache_path)

        print("Constructing User-Item matrix from scratch...")

        # Ensure we only use known users and items
        # (Filter data that might not be in the indexer if indexer was fit on a subset,
        # though standard pipeline fits on all)
        valid_users = train_df["customer_id"].isin(indexer.user_to_idx)
        valid_items = train_df["article_id"].isin(indexer.item_to_idx)
        df_filtered = train_df[valid_users & valid_items].copy()

        # Map IDs to indices
        # Using map is faster than apply for dictionaries
        rows = df_filtered["customer_id"].map(indexer.user_to_idx).astype(np.int32)
        cols = df_filtered["article_id"].map(indexer.item_to_idx).astype(np.int32)

        # Calculate weights: 1 / (days_elapsed + epsilon)
        # Epsilon prevents division by zero if days_elapsed is 0
        weights = 1.0 / (df_filtered["days_elapsed"].values + 1e-5)

        # Construct CSR Matrix
        # Shape: (n_users, n_items)
        n_users = len(indexer.user_to_idx)
        n_items = len(indexer.item_to_idx)

        U = sp.coo_matrix(
            (weights, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
        ).tocsr()

        # Aggregate duplicate entries (sum weights)
        U.sum_duplicates()

        print(f"User-Item Matrix constructed. Shape: {U.shape}, NNZ: {U.nnz}")

        # Cache
        print(f"Caching User-Item matrix to {cache_path}...")
        sp.save_npz(cache_path, U)

        return U

    def build_hybrid_matrix(self, U, articles_df, indexer, load_cached_data=True):
        """
        Constructs the Hybrid Similarity Matrix (S_hybrid).
        Fuses Behavioral Similarity (with IDF) and Variant Similarity (Metadata).

        Args:
            U (sp.csr_matrix): User-Item history matrix.
            articles_df (pd.DataFrame): Articles metadata.
            indexer (Indexer): Object containing mappings.
            load_cached_data (bool): Whether to use cache.

        Returns:
            scipy.sparse.csr_matrix: The hybrid similarity matrix S_hybrid.
        """
        cache_path = os.path.join(self.cache_dir, "S_hybrid.npz")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Hybrid matrix from {cache_path}...")
            return sp.load_npz(cache_path)

        print("Constructing Hybrid Similarity Matrix...")

        # --- 1. Behavioral Similarity (S_behavior) ---
        print("  Computing Behavioral Similarity (TF-IDF + Cosine)...")

        # Apply IDF Weighting
        # IDF = log(N / (DF + 1))
        # N = number of users
        # DF = number of users who purchased item i
        N = U.shape[0]
        # Count non-zeros per column (item)
        DF = np.diff(U.tocsc().indptr)
        IDF = np.log(N / (DF + 1.0))

        # Create diagonal IDF matrix
        IDF_diag = sp.diags(IDF, 0, shape=(U.shape[1], U.shape[1]), format="csr")

        # Apply IDF to U: U_idf = U * IDF_diag
        U_idf = U.dot(IDF_diag)

        # L2 Normalization (Row-wise/User-wise)
        # This ensures power users don't dominate the similarity
        U_norm = normalize(U_idf, norm="l2", axis=1)

        # Compute Item-Item Similarity: S = U_norm.T * U_norm
        # Result is (n_items, n_items)
        # Note: This can be memory intensive.
        # Since we have 220GB RAM, we proceed with sparse multiplication.
        S_behavior = U_norm.T.dot(U_norm)

        # Zero out diagonal (item is always similar to itself, but not useful for recs)
        S_behavior.setdiag(0)

        # --- 2. Variant Similarity (S_variant) ---
        print("  Computing Variant Similarity (Product Code Adjacency)...")
        S_variant = self._build_variant_matrix(articles_df, indexer)

        # --- 3. Fusion ---
        print(
            f"  Fusing matrices with Variant Weight = {self.config.VARIANT_WEIGHT}..."
        )
        # S_hybrid = S_behavior + lambda * S_variant
        # We assume S_variant is already compatible in shape
        S_hybrid = S_behavior + (S_variant * self.config.VARIANT_WEIGHT)

        print(
            f"Hybrid Matrix constructed. Shape: {S_hybrid.shape}, NNZ: {S_hybrid.nnz}"
        )

        # Cache
        print(f"Caching Hybrid matrix to {cache_path}...")
        sp.save_npz(cache_path, S_hybrid)

        return S_hybrid

    def _build_variant_matrix(self, articles_df, indexer):
        """
        Helper to build the sparse adjacency matrix based on product_code.
        Items sharing the same product_code are connected.
        """
        # Filter articles to those in our index
        # We need a map from article_id -> item_idx
        # And article_id -> product_code

        # Create a working dataframe
        df = articles_df[["article_id", "product_code"]].copy()

        # Map article_id to dense item index
        df["item_idx"] = df["article_id"].map(indexer.item_to_idx)

        # Drop articles not in the training set (no index)
        df = df.dropna(subset=["item_idx"])
        df["item_idx"] = df["item_idx"].astype(int)

        # Map product_code to a dense index for matrix construction
        unique_products = df["product_code"].unique()
        prod_map = {code: i for i, code in enumerate(unique_products)}
        df["prod_idx"] = df["product_code"].map(prod_map)

        # Construct P matrix: (n_products, n_items)
        # P[p, i] = 1 if item i belongs to product p
        n_products = len(unique_products)
        n_items = len(indexer.item_to_idx)

        rows = df["prod_idx"].values
        cols = df["item_idx"].values
        data = np.ones(len(rows), dtype=np.float32)

        P = sp.coo_matrix((data, (rows, cols)), shape=(n_products, n_items)).tocsr()

        # S_variant = P.T * P
        # Result: (n_items, n_items). Entry (i, j) is 1 if they share a product code.
        S_variant = P.T.dot(P)

        # Zero out diagonal
        S_variant.setdiag(0)

        return S_variant
