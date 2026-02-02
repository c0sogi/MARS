import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import gc
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

from library.config import Paths, DATA_CONFIG, CANDIDATE_CONFIG, SEED
from library.utils import CacheManager, setup_logger, reduce_mem_usage
from library.data_loader import get_recent_popular_items

logger = setup_logger("retrieval")


class CooccurrenceMatrix:
    """
    Implements a Linear-Decay Item-Item Co-occurrence Matrix.
    Weights are calculated as w = 1 / (1 + days_elapsed).
    """

    def __init__(self):
        self.cache = CacheManager()
        self.matrix: Optional[sp.csr_matrix] = None
        self.article_mapper: Dict[str, int] = {}
        self.reverse_article_mapper: Dict[int, str] = {}

    def fit(self, df: pd.DataFrame, load_cached: bool = True) -> None:
        """
        Builds the item-item co-occurrence matrix from transaction history.

        Args:
            df: Transaction DataFrame containing 'customer_id', 'article_id', 't_dat'.
            load_cached: Whether to attempt loading from cache.
        """
        cache_file_matrix = "cooc_matrix.npz"
        cache_file_map = "cooc_article_map.npy"

        # Try loading from cache
        if (
            load_cached
            and self.cache.exists(cache_file_matrix)
            and self.cache.exists(cache_file_map)
        ):
            logger.info("Loading Cooccurrence Matrix from cache...")
            self.matrix = sp.load_npz(self.cache.get_path(cache_file_matrix))
            article_ids = self.cache.load_npy(cache_file_map)
            self.article_mapper = {aid: i for i, aid in enumerate(article_ids)}
            self.reverse_article_mapper = {i: aid for i, aid in enumerate(article_ids)}
            return

        logger.info("Building Cooccurrence Matrix from scratch...")

        # 1. Filter Data
        window = DATA_CONFIG["cooc_window_days"]
        max_date = df["t_dat"].max()
        cutoff_date = max_date - pd.Timedelta(days=window)

        logger.info(f"Filtering data to last {window} days (>= {cutoff_date})...")
        temp_df = df[df["t_dat"] >= cutoff_date].copy()

        # 2. Calculate Weights (Linear Decay)
        # w = 1 / (diff + 1)
        # We calculate diff relative to the max_date in the dataset
        temp_df["days_elapsed"] = (max_date - temp_df["t_dat"]).dt.days
        temp_df["weight"] = 1.0 / (temp_df["days_elapsed"] + 1.0)

        # 3. Create Mappings
        unique_articles = temp_df["article_id"].unique()
        self.article_mapper = {aid: i for i, aid in enumerate(unique_articles)}
        self.reverse_article_mapper = {i: aid for i, aid in enumerate(unique_articles)}

        # Save mapping for consistency
        self.cache.save_npy(unique_articles, cache_file_map)

        # 4. Construct Bipartite Matrix (Basket x Item)
        # We treat (customer_id, t_dat) as a basket.
        # To save memory and complexity, we can actually just use User x Item if we assume
        # co-occurrence is defined by user history in this window.
        # Given the "Item-Item Co-occurrence" standard definition in this context (items bought together),
        # grouping by user is the standard approach for implicit feedback data.

        logger.info("Constructing sparse User-Item matrix...")
        # Map users to temporary indices
        unique_users = temp_df["customer_id"].unique()
        user_mapper = {uid: i for i, uid in enumerate(unique_users)}

        row_ind = temp_df["customer_id"].map(user_mapper).values
        col_ind = temp_df["article_id"].map(self.article_mapper).values
        data = temp_df["weight"].values

        # Shape: (n_users, n_items)
        user_item_matrix = sp.csc_matrix(
            (data, (row_ind, col_ind)),
            shape=(len(unique_users), len(unique_articles)),
            dtype=np.float32,
        )

        # 5. Compute Co-occurrence: C = U^T * U
        # This computes the dot product of item vectors.
        # If Item A and Item B are bought by the same user, their weights are multiplied and summed.
        logger.info("Computing A^T * A ...")
        self.matrix = user_item_matrix.T.dot(user_item_matrix)

        # Zero out diagonal (item co-occurring with itself is irrelevant)
        self.matrix.setdiag(0)
        self.matrix.eliminate_zeros()

        logger.info(
            f"Matrix built. Shape: {self.matrix.shape}, Non-zeros: {self.matrix.nnz}"
        )

        # 6. Save
        logger.info("Saving Cooccurrence Matrix to cache...")
        sp.save_npz(self.cache.get_path(cache_file_matrix), self.matrix)

        # Cleanup
        del temp_df, user_item_matrix
        gc.collect()

    def predict_batch(
        self, history_df: pd.DataFrame, target_customers: List[str], top_k: int = 12
    ) -> pd.DataFrame:
        """
        Generates candidates for a batch of customers based on their history.
        Uses matrix multiplication: R = History_Matrix * Cooc_Matrix.
        """
        if self.matrix is None:
            raise ValueError("Matrix not fitted. Call fit() first.")

        logger.info(
            f"Predicting co-occurrence candidates for {len(target_customers)} customers..."
        )

        # 1. Build History Matrix for Target Customers
        # Filter history to only target customers and known articles
        valid_articles = set(self.article_mapper.keys())

        # We take the recent history of these customers
        # To respect the decay logic, we should weight recent items higher in the query vector too
        # But for simplicity and speed in retrieval, binary or simple count is often sufficient.
        # Let's use the same decay logic for the query vector to emphasize recent interests.

        mask = history_df["customer_id"].isin(target_customers) & history_df[
            "article_id"
        ].isin(valid_articles)
        subset_df = history_df[mask].copy()

        if subset_df.empty:
            return pd.DataFrame(columns=["customer_id", "article_id"])

        # Calculate weights for query vector
        max_date = subset_df["t_dat"].max()
        subset_df["days_elapsed"] = (max_date - subset_df["t_dat"]).dt.days
        subset_df["weight"] = 1.0 / (subset_df["days_elapsed"] + 1.0)

        # Map IDs
        cust_map = {cid: i for i, cid in enumerate(target_customers)}

        subset_df["user_idx"] = subset_df["customer_id"].map(cust_map)
        subset_df["item_idx"] = subset_df["article_id"].map(self.article_mapper)

        # Create Sparse Query Matrix (Users x Items)
        rows = subset_df["user_idx"].values
        cols = subset_df["item_idx"].values
        data = subset_df["weight"].values

        query_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(target_customers), self.matrix.shape[0])
        )

        # 2. Matrix Multiplication
        # Scores (Users x Items) = Query (Users x Items) * Cooc (Items x Items)
        scores_matrix = query_matrix.dot(self.matrix)

        # 3. Extract Top K
        # We iterate to extract top K per row.
        candidates = []

        # Convert to lil for faster row access or just iterate CSR
        for i in range(len(target_customers)):
            row = scores_matrix[i]
            if row.nnz == 0:
                continue

            # Get indices of top k
            data_row = row.data
            indices_row = row.indices

            if len(data_row) > top_k:
                top_indices_local = np.argpartition(-data_row, top_k)[:top_k]
                best_indices = indices_row[top_indices_local]
                best_scores = data_row[top_indices_local]
            else:
                best_indices = indices_row
                best_scores = data_row

            cust_id = target_customers[i]
            for idx, score in zip(best_indices, best_scores):
                candidates.append(
                    {
                        "customer_id": cust_id,
                        "article_id": self.reverse_article_mapper[idx],
                        "cooc_score": score,
                    }
                )

        return pd.DataFrame(candidates)


class CandidateGenerator:
    """
    Orchestrates the retrieval of candidates from multiple sources:
    1. Graph (LightGCN)
    2. Co-occurrence (Item-Item)
    3. Repurchase (History)
    4. Popularity (Trend)
    """

    def __init__(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        user_map: Dict[str, int],
        item_map: Dict[int, str],
        cooc_matrix: CooccurrenceMatrix,
    ):
        self.u_emb = torch.tensor(user_embeddings, dtype=torch.float32)
        self.i_emb = torch.tensor(item_embeddings, dtype=torch.float32)
        self.user_map = user_map
        self.item_map = item_map  # Maps int index to article_id string
        self.cooc_matrix = cooc_matrix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move item embeddings to GPU once
        self.i_emb = self.i_emb.to(self.device)

    def _get_graph_candidates(
        self, target_customers: List[str], k: int
    ) -> pd.DataFrame:
        """
        Retrieves top-k candidates using LightGCN embeddings (Dot Product).
        """
        logger.info("Generating Graph candidates...")
        valid_customers = [c for c in target_customers if c in self.user_map]
        if not valid_customers:
            return pd.DataFrame(columns=["customer_id", "article_id"])

        # Get indices
        indices = [self.user_map[c] for c in valid_customers]
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        candidates = []
        batch_size = 4096

        # Process in batches to save memory
        with torch.no_grad():
            for i in range(0, len(indices_tensor), batch_size):
                batch_indices = indices_tensor[i : i + batch_size]
                batch_cust_ids = valid_customers[i : i + batch_size]

                # Get User Embeddings [Batch, Dim]
                batch_u_emb = self.u_emb[batch_indices].to(self.device)

                # Score: [Batch, N_Items] = [Batch, Dim] @ [Dim, N_Items]
                scores = torch.matmul(batch_u_emb, self.i_emb.t())

                # Top K
                top_vals, top_indices = torch.topk(scores, k=k, dim=1)
                top_indices = top_indices.cpu().numpy()
                top_vals = top_vals.cpu().numpy()

                # Map back
                for j, cust_id in enumerate(batch_cust_ids):
                    for idx, score in zip(top_indices[j], top_vals[j]):
                        candidates.append(
                            {
                                "customer_id": cust_id,
                                "article_id": self.item_map[idx],
                                "graph_score": score,
                            }
                        )

        return pd.DataFrame(candidates)

    def _get_repurchase_candidates(
        self, history_df: pd.DataFrame, target_customers: List[str], k: int
    ) -> pd.DataFrame:
        """
        Retrieves top-k items from user's own history (Frequency + Recency).
        """
        logger.info("Generating Repurchase candidates...")
        # Filter
        subset = history_df[history_df["customer_id"].isin(target_customers)].copy()

        # We rank by count, then by max date
        # Group by user, item
        agg = (
            subset.groupby(["customer_id", "article_id"])
            .agg(count=("article_id", "count"), last_date=("t_dat", "max"))
            .reset_index()
        )

        # Sort
        agg = agg.sort_values(
            ["customer_id", "count", "last_date"], ascending=[True, False, False]
        )

        # Take top k
        agg = agg.groupby("customer_id").head(k)

        # Rename count to repurchase_score
        agg = agg.rename(columns={"count": "repurchase_score"})

        return agg[["customer_id", "article_id", "repurchase_score"]]

    def generate(
        self,
        target_customers: List[str],
        history_df: pd.DataFrame,
        popular_items: List[str],
    ) -> pd.DataFrame:
        """
        Main method to generate candidates.
        """
        logger.info(
            f"Starting candidate generation for {len(target_customers)} customers..."
        )

        # 1. Graph Candidates
        df_graph = self._get_graph_candidates(
            target_customers, k=CANDIDATE_CONFIG["top_k_graph"]
        )
        logger.info(f"Graph candidates: {len(df_graph)}")

        # 2. Co-occurrence Candidates
        # We use the history_df to find what they bought, then query the matrix
        df_cooc = self.cooc_matrix.predict_batch(
            history_df, target_customers, top_k=CANDIDATE_CONFIG["top_k_cooc"]
        )
        logger.info(f"Co-occurrence candidates: {len(df_cooc)}")

        # 3. Repurchase Candidates
        df_repurchase = self._get_repurchase_candidates(
            history_df, target_customers, k=CANDIDATE_CONFIG["top_k_repurchase"]
        )
        logger.info(f"Repurchase candidates: {len(df_repurchase)}")

        # 4. Popularity (Fallback / Fill)
        # We don't generate a massive DF for popularity for everyone immediately to save memory.
        # We append it to users who have few candidates or just add as a general set.
        # Strategy: Create a base dataframe of popular items for ALL users? No, too big (1.3M * 12).
        # Strategy: Concat known sources, then fill.
        # However, for the ranker, we usually WANT the popular items as negative samples or strong baselines.
        # Let's generate them for everyone.

        # Efficient way: Repeat popular items for all customers
        # This is memory intensive.
        # Optimization: We return the union of the specific strategies.
        # The ranker training usually handles negative sampling separately.
        # BUT, for the test set, we MUST output 12 predictions.
        # So we should ensure every user has at least 12 candidates.

        # Merge specific sources first
        # We use merge/concat to combine scores.
        # Since we want to keep all candidates and merge scores if they exist in multiple,
        # we can concat then groupby max or mean.
        # Max is good for scores.

        combined_df = pd.concat([df_graph, df_cooc, df_repurchase], ignore_index=True)

        # Group by ID and take max of scores (merging rows)
        # This preserves the scores from different sources for the same item
        combined_df = combined_df.groupby(
            ["customer_id", "article_id"], as_index=False
        ).max()

        logger.info("Adding Popularity candidates...")
        # Optimization: Don't use pandas merge/cross. Construct arrays.
        n_cust = len(target_customers)
        n_pop = len(popular_items)

        pop_cust = np.repeat(target_customers, n_pop)
        pop_art = np.tile(popular_items, n_cust)

        df_pop = pd.DataFrame({"customer_id": pop_cust, "article_id": pop_art})
        df_pop["pop_score"] = 1.0  # Indicator for popularity source

        combined_df = pd.concat([combined_df, df_pop], ignore_index=True)

        # Group again to merge popularity
        combined_df = combined_df.groupby(
            ["customer_id", "article_id"], as_index=False
        ).max()

        logger.info(f"Total candidates generated: {len(combined_df)}")

        return combined_df
