import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import gc
import torch
from library import config, data_manager, visual_module


class DualViewRetriever:
    def __init__(self):
        # Load mappings once
        self.cust_to_idx, self.idx_to_cust, self.art_to_idx, self.idx_to_art = (
            data_manager.get_id_mappings()
        )
        self.num_users = len(self.idx_to_cust)
        self.num_items = len(self.idx_to_art)

    def build_sequential_graph(
        self, transactions_df, cache_key="default", load_cached_data=True
    ):
        """
        Constructs the Time-Aware Transition Matrix (T_seq) using Strict Recency.

        Args:
            transactions_df (pd.DataFrame): Transaction history.
            cache_key (str): Unique identifier for caching (e.g., window name).
            load_cached_data (bool): Whether to try loading from disk.

        Returns:
            scipy.sparse.csr_matrix: The sequential transition graph.
        """
        os.makedirs(config.WORKING_DIR, exist_ok=True)
        filename = f"sequential_graph_{cache_key}.npz"
        path = config.WORKING_DIR / filename

        if load_cached_data and os.path.exists(path):
            print(f"Loading sequential graph from {path}...")
            return sp.load_npz(path)

        print(f"Building sequential graph (Key: {cache_key})...")

        # 1. Strict Recency Filter (Last N weeks relative to the data provided)
        # We assume transactions_df contains the relevant history window.
        # We strictly enforce the lookback limit from the max date in this data.
        if len(transactions_df) == 0:
            # Return empty matrix if no data
            return sp.csr_matrix((self.num_items, self.num_items), dtype=np.float32)

        max_date = transactions_df["t_dat"].max()
        cutoff_date = max_date - pd.Timedelta(weeks=config.TRANSITION_HISTORY_WEEKS)

        # Filter data
        df = transactions_df[transactions_df["t_dat"] > cutoff_date].copy()

        if len(df) == 0:
            return sp.csr_matrix((self.num_items, self.num_items), dtype=np.float32)

        # 2. Sort by User and Time
        df = df.sort_values(["customer_id", "t_dat"])

        # 3. Map to Indices
        # We use the pre-computed mappings
        df["article_idx"] = df["article_id"].map(self.art_to_idx)
        # Drop transactions for articles not in our map (rare safety check)
        df = df.dropna(subset=["article_idx"])
        df["article_idx"] = df["article_idx"].astype(np.int32)

        # 4. Generate Transitions (Item A -> Item B)
        # Shift article_idx by -1 within each customer group
        # Vectorized approach: shift column, then check if customer_id changed
        df["next_article_idx"] = df.groupby("customer_id")["article_idx"].shift(-1)

        # Drop the last purchase of each session (no transition)
        pairs = df.dropna(subset=["next_article_idx"])

        # 5. Build Sparse Matrix
        src = pairs["article_idx"].values.astype(np.int32)
        dst = pairs["next_article_idx"].values.astype(np.int32)
        data = np.ones(len(src), dtype=np.float32)

        # Sum duplicates (raw weights)
        t_seq = sp.coo_matrix(
            (data, (src, dst)), shape=(self.num_items, self.num_items)
        ).tocsr()

        print(f"Sequential graph built. Edges: {t_seq.nnz}")
        sp.save_npz(path, t_seq)

        return t_seq

    def build_user_vectors(self, transactions_df, target_customers):
        """
        Constructs User History Vectors (U) with time decay.

        Args:
            transactions_df (pd.DataFrame): User history.
            target_customers (list/array): List of customer_ids to build vectors for.

        Returns:
            scipy.sparse.csr_matrix: Shape (len(target_customers), num_items)
        """
        print(f"Building user vectors for {len(target_customers)} users...")

        # Filter transactions to only the target customers
        # Using a set for faster lookup
        target_cust_set = set(target_customers)
        df = transactions_df[
            transactions_df["customer_id"].isin(target_cust_set)
        ].copy()

        num_targets = len(target_customers)

        if len(df) == 0:
            print(
                "Warning: No history found for target customers. Returning empty vectors."
            )
            return sp.csr_matrix((num_targets, self.num_items), dtype=np.float32)

        # Map customer_id to a local row index (0 to num_targets-1)
        # This aligns the sparse matrix rows with the order of target_customers
        cust_to_local_idx = {cid: i for i, cid in enumerate(target_customers)}
        df["local_idx"] = df["customer_id"].map(cust_to_local_idx)

        # Map article_id to global column index
        df["article_idx"] = df["article_id"].map(self.art_to_idx)

        # Drop invalid
        df = df.dropna(subset=["local_idx", "article_idx"])

        # Calculate Time Decay
        # Weight = 1 / (days_since + 1)
        max_date = df["t_dat"].max()
        df["days_diff"] = (max_date - df["t_dat"]).dt.days
        # Ensure non-negative
        df["days_diff"] = df["days_diff"].clip(lower=0)
        df["weight"] = 1.0 / (df["days_diff"] + 1.0)

        # Build Matrix
        rows = df["local_idx"].values.astype(np.int32)
        cols = df["article_idx"].values.astype(np.int32)
        data = df["weight"].values.astype(np.float32)

        # Use sum aggregation (default for coo_matrix construction with duplicates)
        user_vectors = sp.coo_matrix(
            (data, (rows, cols)), shape=(num_targets, self.num_items)
        ).tocsr()

        return user_vectors

    def retrieve(self, user_vectors, sequential_graph, visual_graph, customer_ids):
        """
        Executes the Dual-View Retrieval: S = U.T_seq + lambda(U.T_vis) + alpha(U)

        Args:
            user_vectors (CSR): User history vectors (U).
            sequential_graph (CSR): Sequential transition matrix (T_seq).
            visual_graph (CSR): Visual similarity graph (T_vis).
            customer_ids (list): List of customer_ids corresponding to rows of user_vectors.

        Returns:
            pd.DataFrame: Candidates with columns [customer_id, article_id, score_seq, score_vis, score_hist]
        """
        print(f"Retrieving candidates for {len(customer_ids)} users...")

        batch_size = 1000
        num_users = user_vectors.shape[0]
        results_list = []

        # Hyperparameters
        lambda_vis = config.LAMBDA_VIS
        alpha_hist = config.ALPHA_HIST
        top_k = config.TOP_K_CANDIDATES

        # Ensure all matrices are CSR for efficient arithmetic
        if not sp.isspmatrix_csr(sequential_graph):
            sequential_graph = sequential_graph.tocsr()
        if not sp.isspmatrix_csr(visual_graph):
            visual_graph = visual_graph.tocsr()

        # Process in batches
        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)

            # Slice batch (CSR slicing is efficient)
            u_batch = user_vectors[start_idx:end_idx]

            # 1. Compute Component Scores (Sparse)
            # S_seq = U @ T_seq
            s_seq = u_batch.dot(sequential_graph)

            # S_vis = U @ T_vis
            s_vis = u_batch.dot(visual_graph)

            # S_hist = U
            s_hist = u_batch

            # 2. Compute Total Score (Dense)
            # We densify the batch to perform Top-K efficiently and accurately.
            # S_total = S_seq + lambda*S_vis + alpha*S_hist
            # Note: Densifying (1000, 100k) is ~400MB, which fits easily in RAM.

            # We compute the weighted sum in sparse format first to minimize dense operations if possible,
            # but converting to dense individually allows us to extract component scores easily.
            # Let's densify components.
            d_seq = s_seq.toarray()
            d_vis = s_vis.toarray()
            d_hist = s_hist.toarray()

            d_total = d_seq + (d_vis * lambda_vis) + (d_hist * alpha_hist)

            # 3. Extract Top-K
            # argpartition to find indices of top K elements
            # If batch has fewer items than K (unlikely), take all
            curr_batch_size = d_total.shape[0]
            n_cols = d_total.shape[1]
            k = min(top_k, n_cols)

            # Get indices of top K values
            # argpartition puts top K at the end
            if k < n_cols:
                top_k_indices = np.argpartition(d_total, -k, axis=1)[:, -k:]
            else:
                top_k_indices = np.argsort(d_total, axis=1)

            # 4. Construct Result DataFrame
            # We iterate through the batch rows to gather data
            batch_results = []

            for i in range(curr_batch_size):
                u_idx = i
                global_cust_id = customer_ids[start_idx + i]

                # Indices for this user
                indices = top_k_indices[i]

                # Sort by total score descending (optional but cleaner)
                scores = d_total[i, indices]
                sort_order = np.argsort(scores)[::-1]
                sorted_indices = indices[sort_order]

                # Extract component scores
                val_seq = d_seq[i, sorted_indices]
                val_vis = d_vis[i, sorted_indices]
                val_hist = d_hist[i, sorted_indices]

                # Map to Article IDs
                art_ids = self.idx_to_art[sorted_indices]

                # Create dicts for DataFrame
                # Using a list of dicts is slower, let's create a small dict of arrays
                user_data = {
                    "customer_id": [global_cust_id] * len(sorted_indices),
                    "article_id": art_ids,
                    "score_seq": val_seq,
                    "score_vis": val_vis,
                    "score_hist": val_hist,
                }
                batch_results.append(pd.DataFrame(user_data))

            # Append batch results
            if batch_results:
                results_list.append(pd.concat(batch_results, ignore_index=True))

            # Memory management
            if start_idx % 5000 == 0:
                gc.collect()

        # Final Concatenation
        if not results_list:
            return pd.DataFrame(
                columns=[
                    "customer_id",
                    "article_id",
                    "score_seq",
                    "score_vis",
                    "score_hist",
                ]
            )

        final_df = pd.concat(results_list, ignore_index=True)
        return final_df
