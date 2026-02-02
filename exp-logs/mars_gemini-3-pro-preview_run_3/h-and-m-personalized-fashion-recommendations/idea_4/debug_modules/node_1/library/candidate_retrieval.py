import os
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from library.config import Config
from library.data_utils import (
    load_customers,
    load_articles,
    load_metadata,
    seed_everything,
)
from library.graph_engine import BehavioralGraphBuilder
from library.visual_engine import VisualGraphBuilder


class CandidateRetriever:
    """
    Implements Stage 1: Multi-View Vectorized Retrieval.
    Generates candidate items for users by propagating their history through
    Behavioral and Visual sparse graphs.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.working_dir = Config.WORKING_DIR
        self.full_history_path = self.working_dir / "full_user_history.npz"

        # Initialize builders
        self.beh_builder = BehavioralGraphBuilder()
        self.vis_builder = VisualGraphBuilder()

        # Load maps
        self.customers_df, self.customer_map = load_customers(load_cached_data=True)
        self.articles_df, self.article_map = load_articles(load_cached_data=True)

        # Reverse maps for decoding results
        self.idx_to_cust = {v: k for k, v in self.customer_map.items()}
        self.idx_to_art = {v: k for k, v in self.article_map.items()}

    def _calculate_time_weights(
        self, dates: pd.Series, reference_date: pd.Timestamp
    ) -> np.ndarray:
        """
        Calculates exponential time decay weights.
        Weight = exp(-decay_rate * days_diff)
        """
        if not np.issubdtype(dates.dtype, np.datetime64):
            dates = pd.to_datetime(dates)

        days_diff = (reference_date - dates).dt.days
        days_diff = days_diff.clip(lower=0)

        weights = np.exp(-Config.TIME_DECAY_RATE * days_diff)
        return weights.values.astype(np.float32)

    def build_full_user_history(
        self, load_cached_data: bool = True
    ) -> sparse.csr_matrix:
        """
        Constructs the User History Matrix (U) using BOTH Train and Validation metadata.
        This ensures we have the most recent history for all users (including those in the
        validation split) to generate predictions for the test set.

        Returns:
            sparse.csr_matrix: Shape (n_customers, n_articles)
        """
        os.makedirs(self.working_dir, exist_ok=True)

        if load_cached_data and self.full_history_path.exists():
            print(f"Loading cached Full User History from {self.full_history_path}")
            return sparse.load_npz(self.full_history_path)

        print("Building Full User History (Train + Val)...")

        # 1. Load Data
        train_df = load_metadata("train")
        val_df = load_metadata("val")

        # Concatenate to get full history
        full_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

        # 2. Map Indices
        full_df["t_dat"] = pd.to_datetime(full_df["t_dat"])
        full_df["article_idx"] = full_df["article_id"].map(self.article_map)
        full_df["customer_idx"] = full_df["customer_id"].map(self.customer_map)

        # Drop invalid mappings (if any)
        full_df = full_df.dropna(subset=["article_idx", "customer_idx"])

        # 3. Calculate Weights
        # Use the absolute max date in the dataset as reference
        max_date = full_df["t_dat"].max()
        weights = self._calculate_time_weights(full_df["t_dat"], max_date)

        # 4. Build Sparse Matrix
        n_customers = len(self.customer_map)
        n_articles = len(self.article_map)

        row_ind = full_df["customer_idx"].values.astype(np.int32)
        col_ind = full_df["article_idx"].values.astype(np.int32)

        # Sum weights for duplicate (user, item) pairs (multiple purchases)
        user_history = sparse.csr_matrix(
            (weights, (row_ind, col_ind)),
            shape=(n_customers, n_articles),
            dtype=np.float32,
        )

        # 5. Save
        print(f"Saving Full User History to {self.full_history_path}")
        sparse.save_npz(self.full_history_path, user_history)

        return user_history

    def retrieve_candidates(
        self,
        customer_indices: np.ndarray,
        load_cached_data: bool = True,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """
        Generates candidates for the specified customers using Multi-View Propagation.

        S = alpha * U_hist + U_hist @ T_trans + lambda * U_hist @ T_vis

        Args:
            customer_indices (np.ndarray): Array of dense customer indices to generate candidates for.
            load_cached_data (bool): Whether to use cached graphs.
            batch_size (int): Number of users to process at once to manage memory.

        Returns:
            pd.DataFrame: DataFrame containing candidates and decomposed scores.
                          Columns: [customer_idx, article_idx, score, score_trans, score_vis, score_hist]
        """
        # 1. Load Graphs
        # T_trans: Behavioral Transition Matrix
        t_trans = self.beh_builder.build_transition_matrix(
            load_cached_data=load_cached_data
        )

        # T_vis: Visual KNN Graph
        t_vis = self.vis_builder.build_knn_graph(load_cached_data=load_cached_data)

        # U: Full User History
        u_hist = self.build_full_user_history(load_cached_data=load_cached_data)

        print(f"Generating candidates for {len(customer_indices)} customers...")

        # Pre-allocate lists for results
        results_cust = []
        results_art = []
        results_score = []
        results_trans = []
        results_vis = []
        results_hist = []

        # Weights
        w_hist = Config.HISTORY_WEIGHT
        w_vis = Config.VISUAL_WEIGHT

        # 2. Batch Processing
        # We iterate through the requested customer indices in batches
        num_batches = int(np.ceil(len(customer_indices) / batch_size))

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(customer_indices))
            batch_cust_idxs = customer_indices[start_idx:end_idx]

            # Slice User History for this batch
            # u_batch shape: (batch_size, n_articles)
            u_batch = u_hist[batch_cust_idxs]

            # Skip empty users (optimization)
            if u_batch.nnz == 0:
                continue

            # --- Multi-View Propagation ---

            # View 1: Repurchase (History)
            # score_hist = u_batch
            # We keep it sparse for now

            # View 2: Behavioral (Transition)
            # score_trans = u_batch @ t_trans
            s_trans_sparse = u_batch.dot(t_trans)

            # View 3: Visual (KNN)
            # score_vis = u_batch @ t_vis
            s_vis_sparse = u_batch.dot(t_vis)

            # --- Aggregation ---
            # To find top-K efficiently, we convert the combined score to dense.
            # S_total = alpha * U + U @ T_trans + lambda * U @ T_vis
            # Note: Converting 1000 x 100,000 to dense float32 is ~400MB. Safe.

            # Convert to dense arrays for summation and sorting
            # We use .toarray() which returns float32/64
            d_hist = u_batch.toarray()
            d_trans = s_trans_sparse.toarray()
            d_vis = s_vis_sparse.toarray()

            # Combined Score
            d_total = (w_hist * d_hist) + d_trans + (w_vis * d_vis)

            # --- Top-K Extraction ---
            k = Config.RETRIEVAL_TOP_K
            n_items = d_total.shape[1]

            # If k >= n_items, take all (unlikely)
            curr_k = min(k, n_items)

            # argpartition puts the top k elements at the end (unsorted)
            # We want indices of top k
            top_k_idx = np.argpartition(d_total, -curr_k, axis=1)[:, -curr_k:]

            # We need to extract the values for these indices for all components
            # Create row indices for fancy indexing
            row_indices = np.arange(len(batch_cust_idxs))[:, None]

            # Extract scores
            top_scores = d_total[row_indices, top_k_idx]
            top_trans = d_trans[row_indices, top_k_idx]
            top_vis = d_vis[row_indices, top_k_idx]
            top_hist = d_hist[row_indices, top_k_idx]

            # Flatten and Store
            # Repeat customer indices for each of their K items
            cust_col = np.repeat(batch_cust_idxs, curr_k)
            art_col = top_k_idx.flatten()

            results_cust.append(cust_col)
            results_art.append(art_col)
            results_score.append(top_scores.flatten())
            results_trans.append(top_trans.flatten())
            results_vis.append(top_vis.flatten())
            results_hist.append(top_hist.flatten())

            # Explicit GC for large dense arrays
            del d_hist, d_trans, d_vis, d_total

        # 3. Construct DataFrame
        print("Constructing Candidate DataFrame...")
        if not results_cust:
            return pd.DataFrame(
                columns=[
                    "customer_idx",
                    "article_idx",
                    "score",
                    "score_trans",
                    "score_vis",
                    "score_hist",
                ]
            )

        df_candidates = pd.DataFrame(
            {
                "customer_idx": np.concatenate(results_cust),
                "article_idx": np.concatenate(results_art),
                "score": np.concatenate(results_score),
                "score_trans": np.concatenate(results_trans),
                "score_vis": np.concatenate(results_vis),
                "score_hist": np.concatenate(results_hist),
            }
        )

        # Filter out zero-score candidates (if any slipped through argpartition on sparse rows)
        df_candidates = df_candidates[df_candidates["score"] > 0]

        # Downcast to save memory
        df_candidates["customer_idx"] = df_candidates["customer_idx"].astype(np.int32)
        df_candidates["article_idx"] = df_candidates["article_idx"].astype(np.int32)
        df_candidates["score"] = df_candidates["score"].astype(np.float32)

        return df_candidates

    def generate_submission_candidates(
        self, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Convenience method to generate candidates for ALL customers in the sample submission.
        """
        # Load test metadata to get target customers
        test_df = load_metadata("test")

        # Map to indices
        # Note: Some test customers might be new (cold start).
        # The customer_map includes all customers from customers.csv, so they should have an index.
        # If they have no history, the retrieval will return 0 scores, effectively empty.
        # The ranker/fallback logic handles cold start (usually via global popularity).
        test_cust_idxs = (
            test_df["customer_id"]
            .map(self.customer_map)
            .dropna()
            .unique()
            .astype(np.int32)
        )

        print(
            f"Retrieving candidates for {len(test_cust_idxs)} submission customers..."
        )
        return self.retrieve_candidates(
            test_cust_idxs, load_cached_data=load_cached_data
        )

    def save_candidates(self, df: pd.DataFrame, filename: str):
        """
        Saves the candidate dataframe to parquet.
        """
        path = self.working_dir / filename
        print(f"Saving candidates to {path}")
        df.to_parquet(path, index=False)
