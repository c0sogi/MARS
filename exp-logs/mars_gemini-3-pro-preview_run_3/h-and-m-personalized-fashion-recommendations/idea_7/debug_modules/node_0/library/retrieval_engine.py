import os
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from datetime import timedelta
from library.config import Config
from library.data_utils import get_id_maps
from library.visual_encoder import build_visual_graph


class DualGraphRetriever:
    """
    Implements the Stage 1 Retrieval logic of the Dual-Graph Vectorized Cascade system.
    Handles construction of Sequential and Visual graphs, and generates candidate
    items via sparse matrix propagation.
    """

    def __init__(self):
        # Load ID mappings to ensure consistent matrix indices
        self.cust_to_idx, self.art_to_idx, self.cust_map, self.art_map = get_id_maps()
        self.num_users = len(self.cust_map)
        self.num_items = len(self.art_map)

    def _compute_time_weights(self, df, ref_date=None):
        """
        Computes time-decay weights for transactions.
        Weight = 1 / (days_elapsed + 1)
        """
        if ref_date is None:
            ref_date = df["t_dat"].max()

        # Calculate days difference
        # Ensure we work with copies to avoid SettingWithCopy warnings on slices
        days_diff = (ref_date - df["t_dat"]).dt.days

        # Simple inverse decay
        weights = 1.0 / (days_diff + 1.0)
        return weights.values

    def build_sequential_graph(self, train_df, load_cached_data=True):
        """
        Constructs the Time-Aware Transition Matrix (T_seq).
        Captures patterns A -> B based on sequential user purchases.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        if load_cached_data and Config.CACHE_SEQUENTIAL_GRAPH.exists():
            print(
                f"Loading cached Sequential Graph from {Config.CACHE_SEQUENTIAL_GRAPH}..."
            )
            return sparse.load_npz(Config.CACHE_SEQUENTIAL_GRAPH)

        print("Building Sequential Graph from scratch...")

        # 1. Filter Data (Strict Recency)
        max_date = train_df["t_dat"].max()
        cutoff_date = max_date - timedelta(weeks=Config.RETRIEVAL_HISTORY_WEEKS)

        df = train_df[train_df["t_dat"] > cutoff_date].copy()

        # 2. Map IDs to indices
        # Filter out unknown articles (if any, though train_df usually defines the universe)
        df = df[df["article_id"].isin(self.art_to_idx)]
        df["art_idx"] = df["article_id"].map(self.art_to_idx).astype(np.int32)

        # 3. Sort by User and Time
        df = df.sort_values(["customer_id", "t_dat"])

        # 4. Create Pairs (A -> B)
        # We shift the article index to get the 'next' item
        df["next_art_idx"] = df.groupby("customer_id")["art_idx"].shift(-1)

        # Drop the last item of each user (no next item)
        pairs = df.dropna(subset=["next_art_idx"])

        if len(pairs) == 0:
            print("Warning: No sequential pairs found. Returning empty graph.")
            t_seq = sparse.csr_matrix(
                (self.num_items, self.num_items), dtype=np.float32
            )
            sparse.save_npz(Config.CACHE_SEQUENTIAL_GRAPH, t_seq)
            return t_seq

        # 5. Calculate Weights
        # We use the weight of the 'source' transaction or just count?
        # Prompt says "retain raw edge weights".
        # We'll use the time weight of the transition to prioritize recent transitions.
        # Using the date of the second item (B) makes sense as it reflects when the transition completed.
        # But using simple counts is also robust. Let's use time-weighted counts.
        weights = self._compute_time_weights(pairs, ref_date=max_date)

        src_indices = pairs["art_idx"].values.astype(np.int32)
        dst_indices = pairs["next_art_idx"].values.astype(np.int32)

        # 6. Construct Sparse Matrix
        # Sum duplicates (multiple users making same transition)
        print(f"Constructing transition matrix with {len(pairs)} pairs...")
        t_seq = sparse.csr_matrix(
            (weights, (src_indices, dst_indices)),
            shape=(self.num_items, self.num_items),
            dtype=np.float32,
        )

        # Save
        print(f"Saving Sequential Graph to {Config.CACHE_SEQUENTIAL_GRAPH}...")
        sparse.save_npz(Config.CACHE_SEQUENTIAL_GRAPH, t_seq)

        return t_seq

    def get_user_history_matrix(self, df, target_customer_ids=None):
        """
        Constructs the User History Matrix (U).
        Rows: Users, Cols: Items. Values: Time-decayed interaction strength.
        """
        # Filter for relevant customers if specified
        if target_customer_ids is not None:
            # Create a set for fast lookup
            target_set = set(target_customer_ids)
            df = df[df["customer_id"].isin(target_set)].copy()
        else:
            df = df.copy()

        if len(df) == 0:
            # Return empty matrix matching target_customer_ids size if provided
            n_rows = (
                len(target_customer_ids)
                if target_customer_ids is not None
                else self.num_users
            )
            return sparse.csr_matrix((n_rows, self.num_items), dtype=np.float32)

        # Map IDs
        # We need to map customer_id to a row index 0..N for the matrix
        # If target_customer_ids is provided, we map specifically to that order
        if target_customer_ids is not None:
            cust_local_map = {cid: i for i, cid in enumerate(target_customer_ids)}
            df = df[df["customer_id"].isin(cust_local_map)]  # Ensure only valid
            row_indices = df["customer_id"].map(cust_local_map).values
            n_rows = len(target_customer_ids)
        else:
            # Use global mapping
            df = df[df["customer_id"].isin(self.cust_to_idx)]
            row_indices = df["customer_id"].map(self.cust_to_idx).values
            n_rows = self.num_users

        # Map Articles
        df = df[df["article_id"].isin(self.art_to_idx)]
        col_indices = df["article_id"].map(self.art_to_idx).values

        # Compute Weights
        weights = self._compute_time_weights(df)

        # Construct CSR
        user_matrix = sparse.csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(n_rows, self.num_items),
            dtype=np.float32,
        )

        return user_matrix

    def _get_global_popularity(self, df, top_k=100):
        """
        Computes top K items by purchase count in the provided dataframe.
        Used as fallback for cold-start users.
        """
        pop_counts = df["article_id"].value_counts().head(top_k)
        pop_ids = pop_counts.index.values
        # Map to indices
        pop_indices = [
            self.art_to_idx[aid] for aid in pop_ids if aid in self.art_to_idx
        ]
        # Pad if necessary (unlikely)
        return np.array(pop_indices)

    def generate_candidates(
        self, history_df, target_customer_ids, load_cached_graphs=True
    ):
        """
        Main Retrieval Function.
        Generates Top-K candidates for target customers using Dual-Graph propagation.

        Args:
            history_df (pd.DataFrame): Transaction history to use for U and T_seq construction.
            target_customer_ids (list/array): List of customer_ids to generate candidates for.
            load_cached_graphs (bool): Whether to load T_seq/T_vis from disk.

        Returns:
            pd.DataFrame: DataFrame with columns [customer_id, article_id, score, rank, ...].
        """
        print(f"Generating candidates for {len(target_customer_ids)} customers...")

        # 1. Load Graphs
        t_seq = self.build_sequential_graph(
            history_df, load_cached_data=load_cached_graphs
        )
        t_vis = build_visual_graph(load_cached_data=load_cached_graphs)

        # 2. Build User History Vector (U) for targets
        # Note: We pass the specific targets to get a matrix of shape (n_targets, n_items)
        u_matrix = self.get_user_history_matrix(history_df, target_customer_ids)

        # 3. Prepare Fallback (Global Popularity)
        # Calculate popularity on the recent history provided
        cutoff_date = history_df["t_dat"].max() - timedelta(weeks=4)
        recent_df = history_df[history_df["t_dat"] > cutoff_date]
        if len(recent_df) == 0:
            recent_df = history_df
        fallback_indices = self._get_global_popularity(
            recent_df, top_k=Config.RETRIEVAL_TOP_K
        )

        # 4. Batch Processing
        batch_size = 1000
        num_targets = len(target_customer_ids)

        results_list = []

        # Pre-allocate arrays for efficiency?
        # Lists are fine for collecting DataFrames/arrays then concatenating.

        for start_idx in range(0, num_targets, batch_size):
            end_idx = min(start_idx + batch_size, num_targets)
            batch_u = u_matrix[start_idx:end_idx]

            # --- Propagation Formula ---
            # S = U * T_seq + lambda * (U * T_vis) + alpha * U

            # 1. Sequential Score
            s_seq = batch_u.dot(t_seq)

            # 2. Visual Score
            # U * T_vis
            s_vis = batch_u.dot(t_vis)

            # 3. Combine
            # We do this in dense format for the batch to allow easy sorting
            # 1000 users * 100k items * 4 bytes = ~400MB. Safe.
            dense_seq = s_seq.toarray()
            dense_vis = s_vis.toarray()
            dense_hist = batch_u.toarray()

            # Total Score
            # Note: We keep components separate for the Ranker features later,
            # but we need a total score for sorting/selection.
            total_score = (
                dense_seq
                + Config.RETRIEVAL_VISUAL_WEIGHT * dense_vis
                + Config.RETRIEVAL_REPURCHASE_WEIGHT * dense_hist
            )

            # --- Selection (Top-K) ---
            k = Config.RETRIEVAL_TOP_K

            # Use argpartition for fast top-k selection (unsorted)
            # We negate score because argpartition puts smallest first
            # We want largest.
            # Alternatively, use argpartition on -total_score

            # Handle case where n_items < k
            curr_k = min(k, self.num_items)

            # Get indices of top k elements
            # axis=1 is per row (user)
            top_k_idx = np.argpartition(-total_score, curr_k - 1, axis=1)[:, :curr_k]

            # Sort these top k (optional for retrieval, but good for rank feature)
            # We can just gather them.

            # Create row indices for fancy indexing
            row_idx = np.arange(len(batch_u))[:, None]

            # Extract scores for these top k
            batch_scores_total = total_score[row_idx, top_k_idx]
            batch_scores_seq = dense_seq[row_idx, top_k_idx]
            batch_scores_vis = dense_vis[row_idx, top_k_idx]
            batch_scores_hist = dense_hist[row_idx, top_k_idx]

            # --- Handle Cold Start / Empty Rows ---
            # If a user has 0 total score (no history, no connections), assign fallback
            row_sums = total_score.sum(axis=1)
            empty_mask = row_sums == 0

            if np.any(empty_mask):
                # Assign fallback indices to these rows
                # We broadcast fallback_indices to the shape of empty rows
                n_empty = np.sum(empty_mask)
                # Ensure fallback fits k
                fb_idx = fallback_indices[:curr_k]
                # Pad if fallback is smaller than k (unlikely)
                if len(fb_idx) < curr_k:
                    fb_idx = np.pad(fb_idx, (0, curr_k - len(fb_idx)), "wrap")

                top_k_idx[empty_mask] = fb_idx
                # Scores for fallback are technically 0 in the personalized sense,
                # but we can leave them as 0 or assign a dummy small value.
                # Leaving as 0 reflects reality for the ranker features.

            # --- Prepare Result Data ---
            # Flatten to long format
            # customer_ids for this batch
            batch_cust_ids = target_customer_ids[start_idx:end_idx]

            # Repeat customer IDs
            long_cust_ids = np.repeat(batch_cust_ids, curr_k)

            # Flatten article indices
            long_art_indices = top_k_idx.flatten()

            # Map article indices back to article_ids
            long_art_ids = self.art_map[long_art_indices]

            # Flatten scores
            long_score_seq = batch_scores_seq.flatten()
            long_score_vis = batch_scores_vis.flatten()
            long_score_hist = batch_scores_hist.flatten()

            # Create DataFrame
            batch_df = pd.DataFrame(
                {
                    "customer_id": long_cust_ids,
                    "article_id": long_art_ids,
                    "score_seq": long_score_seq,
                    "score_vis": long_score_vis,
                    "score_hist": long_score_hist,
                }
            )

            results_list.append(batch_df)

            # Cleanup memory
            del dense_seq, dense_vis, dense_hist, total_score, s_seq, s_vis

        # Concatenate all batches
        candidates_df = pd.concat(results_list, ignore_index=True)

        # Add Rank column (per customer)
        # We sort by a combined score proxy or just trust the order?
        # Since we used argpartition, the order is arbitrary within top-k.
        # Let's compute a total score column and sort.
        candidates_df["total_score"] = (
            candidates_df["score_seq"]
            + Config.RETRIEVAL_VISUAL_WEIGHT * candidates_df["score_vis"]
            + Config.RETRIEVAL_REPURCHASE_WEIGHT * candidates_df["score_hist"]
        )

        candidates_df = candidates_df.sort_values(
            ["customer_id", "total_score"], ascending=[True, False]
        )
        candidates_df["rank"] = candidates_df.groupby("customer_id").cumcount() + 1

        # Clean up
        gc.collect()

        return candidates_df
