import os
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import timedelta
from library import config, data_loader, graph_builder


class SparseRetriever:
    """
    Stage 1: Multi-Modal Vectorized Retrieval.
    Implements the retrieval logic using sparse matrix propagation.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the retriever by loading graphs and mappings.
        """
        self.load_cached_data = load_cached_data

        # 1. Load Mappings (Must match graph_builder logic)
        print("Loading global mappings...")
        self.cust_id_to_idx, self.cust_idx_to_id = self._get_customer_map()
        self.art_id_to_idx, self.art_idx_to_id = self._get_article_map()
        self.n_users = len(self.cust_id_to_idx)
        self.n_items = len(self.art_id_to_idx)

        # 2. Load Graphs
        print("Loading sparse graphs...")
        self.T_seq = graph_builder.build_sequential_graph(load_cached_data)
        self.T_vis = graph_builder.build_visual_graph(load_cached_data)

        # We load the raw history for the repurchase signal (alpha term)
        self.U_history_raw = graph_builder.build_user_history(load_cached_data)

        # 3. Build/Load Weighted User History (User Representation for Propagation)
        # This includes Time-Decay as per "Importance-Based Aggregation"
        print("Loading/Building weighted user history...")
        self.U_weighted = self._build_weighted_user_history(load_cached_data)

        # 4. Compute Global Popularity (Fallback for Cold Start)
        self.global_popularity = self._compute_global_popularity()

    def _get_customer_map(self):
        """Replicates the mapping logic from graph_builder."""
        customers_df = data_loader.load_customers(self.load_cached_data)
        unique_ids = np.sort(customers_df["customer_id"].unique())
        id_to_idx = {cid: i for i, cid in enumerate(unique_ids)}
        return id_to_idx, unique_ids

    def _get_article_map(self):
        """Replicates the mapping logic from graph_builder."""
        articles_df = data_loader.load_articles(self.load_cached_data)
        unique_ids = np.sort(articles_df["article_id"].unique())
        id_to_idx = {aid: i for i, aid in enumerate(unique_ids)}
        return id_to_idx, unique_ids

    def _build_weighted_user_history(self, load_cached_data):
        """
        Constructs a sparse user matrix with time-decayed weights.
        Weight = exp(-days_elapsed / decay_rate).
        """
        cache_path = config.WORKING_DIR / "weighted_user_history.npz"

        if load_cached_data and cache_path.exists():
            print(f"Loading cached weighted history from {cache_path}...")
            return sp.load_npz(cache_path)

        print("Building weighted user history from scratch...")

        # Load transactions
        df = data_loader.load_transactions("train", load_cached_data)

        # Filter valid items/users
        df = df[
            df["customer_id"].isin(self.cust_id_to_idx)
            & df["article_id"].isin(self.art_id_to_idx)
        ].copy()

        # Calculate Weights
        max_date = df["t_dat"].max()
        # Calculate days difference
        df["days_diff"] = (max_date - df["t_dat"]).dt.days
        # Apply exponential decay
        # config.TIME_DECAY_DAYS is the half-life or scale.
        # Using standard decay: exp(-t / tau)
        df["weight"] = np.exp(-df["days_diff"] / config.TIME_DECAY_DAYS)

        # Map to indices
        row_idx = df["customer_id"].map(self.cust_id_to_idx).astype(np.int32).values
        col_idx = df["article_id"].map(self.art_id_to_idx).astype(np.int32).values
        weights = df["weight"].astype(np.float32).values

        # Build Matrix (Sum weights for multiple interactions)
        matrix = sp.coo_matrix(
            (weights, (row_idx, col_idx)), shape=(self.n_users, self.n_items)
        ).tocsr()

        print(f"Saving weighted history to {cache_path}...")
        sp.save_npz(cache_path, matrix)

        del df, row_idx, col_idx, weights
        gc.collect()

        return matrix

    def _compute_global_popularity(self):
        """Computes top items by weighted frequency for cold-start fallback."""
        # Sum columns of weighted history to get popularity score
        # Convert to dense array (1 x n_items)
        pop_scores = np.array(self.U_weighted.sum(axis=0)).flatten()

        # Get top K indices
        top_k = config.RETRIEVAL_TOP_K
        if len(pop_scores) < top_k:
            top_k = len(pop_scores)

        # argpartition for efficiency
        top_indices = np.argpartition(pop_scores, -top_k)[-top_k:]
        # Sort by score descending
        top_indices = top_indices[np.argsort(pop_scores[top_indices])[::-1]]

        return self.art_idx_to_id[top_indices]

    def propagate(self, user_indices):
        """
        Computes retrieval scores for a batch of users.
        S = U_w @ T_seq + lambda * (U_w @ T_vis) + alpha * U_raw

        Args:
            user_indices (np.array): Array of global user indices.

        Returns:
            sp.csr_matrix: Sparse score matrix (Batch x Items).
        """
        # Slice User Representations
        # U_batch is (Batch x Items)
        u_weighted_batch = self.U_weighted[user_indices]
        u_raw_batch = self.U_history_raw[user_indices]

        # 1. Sequential Signal: U @ T_seq
        # Result is (Batch x Items)
        s_seq = u_weighted_batch.dot(self.T_seq)

        # 2. Visual Signal: U @ T_vis
        # Result is (Batch x Items)
        s_vis = u_weighted_batch.dot(self.T_vis)

        # 3. Combine Scores
        # S = Seq + Lambda * Vis + Alpha * History
        # We use linear combination.
        # Note: Matrices are sparse, operations are efficient.

        # Weighted sum
        scores = (
            s_seq
            + (config.LAMBDA_VISUAL * s_vis)
            + (config.ALPHA_HISTORY * u_raw_batch)
        )

        return scores

    def generate_candidates(self, customer_ids, batch_size=1000):
        """
        Generates candidate items for the given customers.

        Args:
            customer_ids (list): List of customer_id strings.
            batch_size (int): Number of users to process at once.

        Returns:
            dict: {customer_id: [article_id, ...]}
        """
        results = {}

        # Filter valid customers
        valid_customers = []
        valid_indices = []

        for cid in customer_ids:
            if cid in self.cust_id_to_idx:
                valid_customers.append(cid)
                valid_indices.append(self.cust_id_to_idx[cid])
            else:
                # Cold start (totally new user not in training)
                # Assign global popularity immediately
                results[cid] = self.global_popularity[: config.RETRIEVAL_TOP_K].tolist()

        n_valid = len(valid_indices)
        print(
            f"Generating candidates for {n_valid} valid users ({len(customer_ids) - n_valid} cold start)..."
        )

        # Process in batches
        for i in range(0, n_valid, batch_size):
            end = min(i + batch_size, n_valid)
            batch_indices = valid_indices[i:end]
            batch_cids = valid_customers[i:end]

            # Get Scores (Sparse)
            scores_sparse = self.propagate(batch_indices)

            # Densify for Top-K extraction
            # Shape: (Batch, N_Items). N_Items ~ 105k. Batch=1000 -> ~400MB. Safe.
            scores_dense = scores_sparse.toarray()

            # Extract Top K per user
            k = config.RETRIEVAL_TOP_K

            # Vectorized Top-K using argpartition
            # We want indices of top K elements
            # If a user has fewer than K non-zero scores, this still works (zeros included)

            # argpartition puts top K elements at the end
            if scores_dense.shape[1] >= k:
                top_k_idx = np.argpartition(scores_dense, -k, axis=1)[:, -k:]
            else:
                top_k_idx = np.argsort(scores_dense, axis=1)

            # Sort the top K for ranking (optional for retrieval, but good for debugging)
            # We need to gather the values to sort
            rows = np.arange(scores_dense.shape[0])[:, None]
            top_k_scores = scores_dense[rows, top_k_idx]

            # Sort indices by score descending
            sort_order = np.argsort(top_k_scores, axis=1)[:, ::-1]
            sorted_top_k_idx = top_k_idx[rows, sort_order]

            # Map back to Article IDs
            for j, cid in enumerate(batch_cids):
                # Get indices for this user
                idx_list = sorted_top_k_idx[j]

                # Check if scores are zero (Cold start within training set - inactive users)
                # If the max score is 0, the user has no relevant history/propagation
                if top_k_scores[j].max() <= 0:
                    results[cid] = self.global_popularity[:k].tolist()
                else:
                    # Map to IDs
                    art_ids = self.art_idx_to_id[idx_list]
                    results[cid] = art_ids.tolist()

            if (i // batch_size) % 10 == 0:
                print(f"Processed {end}/{n_valid} users...")
                gc.collect()

        return results
