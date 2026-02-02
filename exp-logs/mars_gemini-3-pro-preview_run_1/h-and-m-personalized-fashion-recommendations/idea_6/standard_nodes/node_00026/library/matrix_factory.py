import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from sklearn.preprocessing import normalize
from library.config import Config


class MatrixFactory:
    """
    Factory class for constructing sparse matrices used in the SDCC model.
    Handles caching, time-decay weighting, IDF weighting, and transition counting.
    """

    @staticmethod
    def build_user_history_matrix(df, user_to_idx, item_to_idx, load_cached_data=True):
        """
        Builds the time-decayed user history matrix (U_decayed).
        Weights are calculated as: weight = 1 / (days_elapsed + 1) ** ALPHA.

        Args:
            df (pd.DataFrame): Transaction DataFrame containing 'customer_id', 'article_id', 't_dat'.
            user_to_idx (dict): Mapping from customer_id to user index.
            item_to_idx (dict): Mapping from article_id to item index.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            sp.csr_matrix: Sparse matrix of shape (n_users, n_items).
        """
        cache_path = os.path.join(Config.CACHE_DIR, "user_history_matrix.npz")

        if load_cached_data and os.path.exists(cache_path):
            print("Loading user history matrix from cache...")
            return sp.load_npz(cache_path)

        print("Building user history matrix from scratch...")

        # Ensure t_dat is datetime
        if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
            df = df.copy()
            df["t_dat"] = pd.to_datetime(df["t_dat"])

        max_date = df["t_dat"].max()

        # Calculate days elapsed
        days_elapsed = (max_date - df["t_dat"]).dt.days.values

        # Calculate time decay weights
        weights = 1.0 / np.power(days_elapsed + 1.0, Config.TIME_DECAY_ALPHA)
        weights = weights.astype(np.float32)

        # Map IDs to indices
        # We use map with fillna(-1) to handle potential missing keys safely,
        # though strictly all should be present given the mapping logic.
        user_indices = df["customer_id"].map(user_to_idx).fillna(-1).astype(np.int32)
        item_indices = df["article_id"].map(item_to_idx).fillna(-1).astype(np.int32)

        # Filter out invalid indices (if any)
        mask = (user_indices != -1) & (item_indices != -1)
        user_indices = user_indices[mask]
        item_indices = item_indices[mask]
        weights = weights[mask]

        n_users = len(user_to_idx)
        n_items = len(item_to_idx)

        # Construct CSR matrix
        # Duplicate (user, item) entries are summed by default in CSR construction
        matrix = sp.csr_matrix(
            (weights, (user_indices, item_indices)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        print(f"User History Matrix built. Shape: {matrix.shape}, NNZ: {matrix.nnz}")
        print(f"Saving to {cache_path}...")
        sp.save_npz(cache_path, matrix)

        return matrix

    @staticmethod
    def build_symmetric_similarity(df, user_to_idx, item_to_idx, load_cached_data=True):
        """
        Builds the symmetric item-item similarity matrix (S_sym).
        Logic: S_sym = X_norm^T * X_norm
        Where X is an IDF-weighted, Row-normalized binary user-item matrix.

        Args:
            df (pd.DataFrame): Transaction DataFrame.
            user_to_idx (dict): User mapping.
            item_to_idx (dict): Item mapping.
            load_cached_data (bool): Cache flag.

        Returns:
            sp.csr_matrix: Sparse similarity matrix of shape (n_items, n_items).
        """
        cache_path = os.path.join(Config.CACHE_DIR, "symmetric_similarity.npz")

        if load_cached_data and os.path.exists(cache_path):
            print("Loading symmetric similarity matrix from cache...")
            return sp.load_npz(cache_path)

        print("Building symmetric similarity matrix...")

        # Filter noise: Remove items with low purchase count for similarity calculation
        item_counts = df["article_id"].value_counts()
        valid_items = item_counts[item_counts >= Config.MIN_ITEM_PURCHASES].index
        df_filtered = df[df["article_id"].isin(valid_items)].copy()

        print(
            f"Filtered transactions for similarity: {len(df_filtered)} (Original: {len(df)})"
        )

        # Map indices
        user_indices = (
            df_filtered["customer_id"].map(user_to_idx).fillna(-1).astype(np.int32)
        )
        item_indices = (
            df_filtered["article_id"].map(item_to_idx).fillna(-1).astype(np.int32)
        )

        mask = (user_indices != -1) & (item_indices != -1)
        user_indices = user_indices[mask]
        item_indices = item_indices[mask]

        n_users = len(user_to_idx)
        n_items = len(item_to_idx)

        # 1. Build Binary User-Item Matrix (X)
        # We use binary presence (1) for structure
        ones = np.ones(len(user_indices), dtype=np.float32)
        X = sp.csr_matrix(
            (ones, (user_indices, item_indices)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        # Ensure binary (if user bought item multiple times, make it 1)
        X.data = np.ones_like(X.data)

        # 2. IDF Weighting
        # IDF_i = log(N_users / freq_i)
        print("Calculating IDF weights...")
        col_counts = np.array(X.sum(axis=0)).flatten()
        col_counts[col_counts == 0] = 1  # Avoid div by zero
        idf = np.log(n_users / col_counts).astype(np.float32)

        # Apply IDF to columns of X
        # X * Diag(IDF)
        idf_diag = sp.diags(
            idf, offsets=0, shape=(n_items, n_items), format="csr", dtype=np.float32
        )
        X_idf = X.dot(idf_diag)

        # 3. Row-wise Normalization (L2)
        print("Normalizing rows...")
        X_norm = normalize(X_idf, norm="l2", axis=1)

        # 4. Compute S_sym = X^T X
        print("Computing X^T X (Dot Product)...")
        S_sym = X_norm.T.dot(X_norm)

        # Cast to float32 to ensure precision consistency
        S_sym = S_sym.astype(np.float32)

        print(f"Symmetric Matrix built. Shape: {S_sym.shape}, NNZ: {S_sym.nnz}")
        print(f"Saving to {cache_path}...")
        sp.save_npz(cache_path, S_sym)

        return S_sym

    @staticmethod
    def build_transition_matrix(df, user_to_idx, item_to_idx, load_cached_data=True):
        """
        Builds the directed forward transition matrix (S_fwd).
        S_ij = count(i -> j)
        Normalized column-wise to unit probability.

        Args:
            df (pd.DataFrame): Transaction DataFrame.
            user_to_idx (dict): User mapping.
            item_to_idx (dict): Item mapping.
            load_cached_data (bool): Cache flag.

        Returns:
            sp.csr_matrix: Sparse transition matrix of shape (n_items, n_items).
        """
        cache_path = os.path.join(Config.CACHE_DIR, "transition_matrix.npz")

        if load_cached_data and os.path.exists(cache_path):
            print("Loading transition matrix from cache...")
            return sp.load_npz(cache_path)

        print("Building transition matrix...")

        # Filter noise first
        item_counts = df["article_id"].value_counts()
        valid_items = item_counts[item_counts >= Config.MIN_ITEM_PURCHASES].index
        df_filtered = df[df["article_id"].isin(valid_items)].copy()

        # Sort by user and time to identify sequences
        df_sorted = df_filtered.sort_values(["customer_id", "t_dat"])

        # Map IDs
        user_indices = (
            df_sorted["customer_id"].map(user_to_idx).fillna(-1).astype(np.int32)
        )
        item_indices = (
            df_sorted["article_id"].map(item_to_idx).fillna(-1).astype(np.int32)
        )

        # Create temp DF for shifting
        temp = pd.DataFrame({"user_idx": user_indices, "item_idx": item_indices})

        # Remove invalid mappings
        temp = temp[(temp["user_idx"] != -1) & (temp["item_idx"] != -1)]

        # Shift to get next item
        temp["next_user_idx"] = temp["user_idx"].shift(-1)
        temp["next_item_idx"] = temp["item_idx"].shift(-1)

        # Filter for valid transitions:
        # 1. Same user (user_idx == next_user_idx)
        # 2. Not NaN (last row)
        valid_transitions = temp[temp["user_idx"] == temp["next_user_idx"]].dropna()

        # Count transitions (i -> j)
        print("Counting transitions...")
        counts = (
            valid_transitions.groupby(["item_idx", "next_item_idx"])
            .size()
            .reset_index(name="count")
        )

        row = counts["item_idx"].values
        col = counts["next_item_idx"].values
        data = counts["count"].values.astype(np.float32)

        n_items = len(item_to_idx)

        S_fwd = sp.csr_matrix(
            (data, (row, col)), shape=(n_items, n_items), dtype=np.float32
        )

        # Normalize columns to unit probability (axis=0)
        print("Normalizing columns...")
        S_fwd = normalize(S_fwd, norm="l1", axis=0)

        print(f"Transition Matrix built. Shape: {S_fwd.shape}, NNZ: {S_fwd.nnz}")
        print(f"Saving to {cache_path}...")
        sp.save_npz(cache_path, S_fwd)

        return S_fwd

    @staticmethod
    def get_hybrid_matrix(s_sym, s_fwd):
        """
        Combines the symmetric and forward matrices.
        S_hybrid = S_sym + (LAMBDA_FWD * S_fwd)

        Args:
            s_sym (sp.csr_matrix): Symmetric similarity matrix.
            s_fwd (sp.csr_matrix): Forward transition matrix.

        Returns:
            sp.csr_matrix: Combined hybrid matrix.
        """
        print(f"Combining matrices (Lambda={Config.LAMBDA_FWD})...")

        if s_sym.shape != s_fwd.shape:
            raise ValueError(
                f"Shape mismatch: S_sym {s_sym.shape} vs S_fwd {s_fwd.shape}"
            )

        # Linear combination
        S_hybrid = s_sym + (Config.LAMBDA_FWD * s_fwd)

        return S_hybrid
