import os
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from datetime import timedelta
from library.config import Config
from library.data_utils import (
    load_metadata,
    load_customers,
    load_articles,
    seed_everything,
)
from library.graph_engine import BehavioralGraphBuilder
from library.visual_engine import VisualGraphBuilder


class RankerDatasetGenerator:
    """
    Generates training and validation datasets for the Ranking Stage (LightGBM).
    Implements a Sliding Window strategy to create (History -> Target) samples
    and enriches them with features.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.working_dir = Config.WORKING_DIR

        # Load static metadata once
        self.customers_df, self.customer_map = load_customers(load_cached_data=True)
        self.articles_df, self.article_map = load_articles(load_cached_data=True)

        # Load Graph Builders (for accessing global matrices)
        self.beh_builder = BehavioralGraphBuilder()
        self.vis_builder = VisualGraphBuilder()

    def _calculate_time_weights(
        self, dates: pd.Series, reference_date: pd.Timestamp
    ) -> np.ndarray:
        """
        Calculates exponential time decay weights.
        """
        days_diff = (reference_date - dates).dt.days
        days_diff = days_diff.clip(lower=0)
        return np.exp(-Config.TIME_DECAY_RATE * days_diff).values.astype(np.float32)

    def _build_local_user_history(
        self, history_df: pd.DataFrame, target_users: np.ndarray
    ) -> sparse.csr_matrix:
        """
        Builds a User History matrix (U) restricted to the provided history_df.
        Only includes rows for target_users to save memory.
        """
        # Filter history for relevant users
        # We need to map target_users (dense idx) back to raw IDs or just filter by dense idx if available
        # history_df should have 'customer_idx' and 'article_idx'

        # Optimization: We assume history_df is already filtered by date.
        # We now filter by user set.
        mask = history_df["customer_idx"].isin(target_users)
        relevant_df = history_df[mask].copy()

        if relevant_df.empty:
            return sparse.csr_matrix(
                (len(self.customer_map), len(self.article_map)), dtype=np.float32
            )

        # Calculate weights
        max_date = relevant_df["t_dat"].max()
        weights = self._calculate_time_weights(relevant_df["t_dat"], max_date)

        # Build Sparse Matrix
        # We want the matrix to be shape (n_customers, n_articles)
        # But to save memory, we could build it only for target_users.
        # However, to keep indices aligned with global T matrices, it's safer to build full shape
        # or use slicing. Building full shape is easier for dot product alignment.

        row_ind = relevant_df["customer_idx"].values
        col_ind = relevant_df["article_idx"].values

        n_customers = len(self.customer_map)
        n_articles = len(self.article_map)

        U = sparse.csr_matrix(
            (weights, (row_ind, col_ind)),
            shape=(n_customers, n_articles),
            dtype=np.float32,
        )

        return U

    def _generate_candidates_for_window(
        self,
        history_df: pd.DataFrame,
        target_users: np.ndarray,
        t_trans: sparse.csr_matrix,
        t_vis: sparse.csr_matrix,
    ) -> pd.DataFrame:
        """
        Generates candidates for a specific time window using Multi-View Propagation.
        """
        # 1. Build Local User History
        U_local = self._build_local_user_history(history_df, target_users)

        # Slice U to get only target users for efficient multiplication
        # This reduces the matrix from (1.3M, N) to (Batch, N)
        U_batch = U_local[target_users]

        if U_batch.nnz == 0:
            return pd.DataFrame()

        # 2. Propagation
        # View 1: History
        # View 2: Transition
        S_trans = U_batch.dot(t_trans)
        # View 3: Visual
        S_vis = U_batch.dot(t_vis)

        # 3. Aggregation
        # Convert to dense for top-k selection
        # Note: Doing this in batches is safer if target_users is very large
        # We'll implement a mini-batch loop here

        batch_size = 1000
        results_list = []

        # Weights
        w_vis = Config.VISUAL_WEIGHT
        w_hist = Config.HISTORY_WEIGHT

        num_users = len(target_users)

        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            batch_idxs = np.arange(start, end)  # Indices relative to U_batch

            # Extract dense blocks
            d_hist = U_batch[start:end].toarray()
            d_trans = S_trans[start:end].toarray()
            d_vis = S_vis[start:end].toarray()

            # Combined Score
            d_total = (w_hist * d_hist) + d_trans + (w_vis * d_vis)

            # Top-K
            k = Config.RETRIEVAL_TOP_K
            curr_k = min(k, d_total.shape[1])

            # Indices of top k
            top_k_idx = np.argpartition(d_total, -curr_k, axis=1)[:, -curr_k:]

            # Extract scores
            row_indices = np.arange(len(batch_idxs))[:, None]

            top_scores = d_total[row_indices, top_k_idx]
            top_trans = d_trans[row_indices, top_k_idx]
            top_vis = d_vis[row_indices, top_k_idx]
            top_hist = d_hist[row_indices, top_k_idx]

            # Map back to global customer indices
            global_cust_idxs = target_users[start:end]

            # Create DataFrame block
            cust_col = np.repeat(global_cust_idxs, curr_k)
            art_col = top_k_idx.flatten()

            batch_df = pd.DataFrame(
                {
                    "customer_idx": cust_col,
                    "article_idx": art_col,
                    "score": top_scores.flatten(),
                    "score_trans": top_trans.flatten(),
                    "score_vis": top_vis.flatten(),
                    "score_hist": top_hist.flatten(),
                }
            )

            # Filter zero scores
            batch_df = batch_df[batch_df["score"] > 0]
            results_list.append(batch_df)

        if not results_list:
            return pd.DataFrame()

        return pd.concat(results_list, ignore_index=True)

    def construct_features(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches the candidate DataFrame with User and Article features.
        """
        print("Constructing features...")

        # 1. Merge Customer Features
        # Select relevant columns
        cust_cols = [
            "customer_idx",
            "age",
            "club_member_status_idx",
            "fashion_news_frequency_idx",
        ]
        candidates_df = candidates_df.merge(
            self.customers_df[cust_cols], on="customer_idx", how="left"
        )

        # 2. Merge Article Features
        art_cols = [
            "article_idx",
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "department_no",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]
        candidates_df = candidates_df.merge(
            self.articles_df[art_cols], on="article_idx", how="left"
        )

        # 3. Merge Global Popularity
        # We use the cached global popularity file
        pop_df = self.beh_builder.build_global_popularity(load_cached_data=True)
        candidates_df = candidates_df.merge(
            pop_df[["article_idx", "global_popularity"]], on="article_idx", how="left"
        )
        candidates_df["global_popularity"] = candidates_df["global_popularity"].fillna(
            0
        )

        return candidates_df

    def generate_sliding_window_data(self, load_cached_data: bool = True):
        """
        Main method to generate Train and Validation datasets using sliding windows.
        """
        os.makedirs(self.working_dir, exist_ok=True)

        if (
            load_cached_data
            and Config.RANKER_TRAIN_SET.exists()
            and Config.RANKER_VAL_SET.exists()
        ):
            print("Loading cached Ranker Datasets...")
            return  # Files exist, nothing to do

        print("Generating Sliding Window Data for Ranker...")

        # 1. Load All Transactions (Train + Val)
        train_df = load_metadata("train")
        val_df = load_metadata("val")
        full_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

        # Preprocess
        full_df["t_dat"] = pd.to_datetime(full_df["t_dat"])
        full_df["article_idx"] = full_df["article_id"].map(self.article_map)
        full_df["customer_idx"] = full_df["customer_id"].map(self.customer_map)
        full_df = full_df.dropna(subset=["article_idx", "customer_idx"])
        full_df["article_idx"] = full_df["article_idx"].astype(np.int32)
        full_df["customer_idx"] = full_df["customer_idx"].astype(np.int32)

        # 2. Load Global Graphs
        print("Loading Global Graphs...")
        t_trans = self.beh_builder.build_transition_matrix(load_cached_data=True)
        t_vis = self.vis_builder.build_knn_graph(load_cached_data=True)

        # 3. Define Windows
        # Latest date
        max_date = full_df["t_dat"].max()

        train_dfs = []
        val_dfs = []

        # We iterate backwards
        # Window 0 is the Validation Set (Last week)
        # Windows 1 to N are Training Sets

        total_weeks = Config.SLIDING_WINDOW_WEEKS

        for w in range(total_weeks):
            print(f"Processing Window {w+1}/{total_weeks}...")

            # Define Time Boundaries
            window_end = max_date - timedelta(days=7 * w)
            window_start = window_end - timedelta(days=6)  # 7 days inclusive

            history_cutoff = window_start - timedelta(days=1)

            print(f"  Target Week: {window_start.date()} to {window_end.date()}")
            print(f"  History Cutoff: {history_cutoff.date()}")

            # Identify Target Users (Users who bought something in this week)
            target_mask = (full_df["t_dat"] >= window_start) & (
                full_df["t_dat"] <= window_end
            )
            target_transactions = full_df[target_mask]

            target_users = target_transactions["customer_idx"].unique()
            print(f"  Active Users in Target: {len(target_users)}")

            if len(target_users) == 0:
                continue

            # History Data
            history_mask = full_df["t_dat"] <= history_cutoff
            history_df = full_df[history_mask]

            # Generate Candidates
            candidates = self._generate_candidates_for_window(
                history_df, target_users, t_trans, t_vis
            )

            if candidates.empty:
                continue

            # Label Candidates
            # Create a set of (user, item) tuples for ground truth
            # Optimization: Use a merged dataframe to check existence
            gt_df = target_transactions[
                ["customer_idx", "article_idx"]
            ].drop_duplicates()
            gt_df["label"] = 1

            candidates = candidates.merge(
                gt_df, on=["customer_idx", "article_idx"], how="left"
            )
            candidates["label"] = candidates["label"].fillna(0).astype(np.int8)

            # Add Week ID (for potential group-wise CV, though we split by time)
            candidates["week"] = w

            # Store
            if w < Config.VAL_WEEKS:
                val_dfs.append(candidates)
            else:
                train_dfs.append(candidates)

            # GC
            del candidates, history_df, target_transactions, gt_df
            gc.collect()

        # 4. Concatenate and Enrich
        print("Concatenating and Enriching Datasets...")

        if train_dfs:
            full_train = pd.concat(train_dfs, ignore_index=True)
            full_train = self.construct_features(full_train)
            print(f"Saving Train Set: {len(full_train)} rows")
            full_train.to_parquet(Config.RANKER_TRAIN_SET, index=False)
            del full_train

        if val_dfs:
            full_val = pd.concat(val_dfs, ignore_index=True)
            full_val = self.construct_features(full_val)
            print(f"Saving Val Set: {len(full_val)} rows")
            full_val.to_parquet(Config.RANKER_VAL_SET, index=False)
            del full_val

        print("Dataset Generation Complete.")
