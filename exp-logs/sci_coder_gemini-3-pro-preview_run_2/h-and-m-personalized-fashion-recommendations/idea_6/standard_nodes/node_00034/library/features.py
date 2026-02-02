import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from library.config import Paths, SEED
from library.utils import setup_logger, reduce_mem_usage, CacheManager

logger = setup_logger("feature_engineering")


class FeatureEngineer:
    """
    Computes interaction-aware features for the ranking stage.
    """

    def __init__(self):
        self.cache = CacheManager()

    def _compute_sales_velocity(
        self, transactions: pd.DataFrame, days: int = 14
    ) -> pd.DataFrame:
        """
        Calculates the slope of daily sales for each article over the last N days.
        Positive slope = Trending Up.
        """
        logger.info(f"Computing sales velocity (trend) over last {days} days...")

        # Filter to recent data
        max_date = transactions["t_dat"].max()
        start_date = max_date - pd.Timedelta(days=days)
        recent = transactions[transactions["t_dat"] > start_date].copy()

        # Count daily sales per article
        daily_counts = (
            recent.groupby(["article_id", "t_dat"]).size().reset_index(name="count")
        )

        # Convert date to integer day index (0 to days-1)
        daily_counts["day_idx"] = (daily_counts["t_dat"] - start_date).dt.days

        # We want slope of count vs day_idx for each article
        # Slope formula: sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
        # Since x (days) is fixed for the window, we can optimize.
        # However, data is sparse (missing days for some items).
        # We will use a simplified approach: aggregation.

        # Pivot to fill zeros: index=article_id, columns=day_idx, values=count
        pivot = daily_counts.pivot_table(
            index="article_id", columns="day_idx", values="count", fill_value=0
        )

        # X values are column indices
        x = pivot.columns.values.astype(float)
        y = pivot.values

        # Vectorized linear regression slope
        # slope = (N * sum(xy) - sum(x)sum(y)) / (N * sum(x^2) - (sum(x))^2)
        N = len(x)
        sum_x = np.sum(x)
        sum_x2 = np.sum(x**2)
        sum_y = np.sum(y, axis=1)
        sum_xy = np.sum(y * x, axis=1)

        denominator = N * sum_x2 - sum_x**2
        if denominator == 0:
            slopes = np.zeros(len(y))
        else:
            slopes = (N * sum_xy - sum_x * sum_y) / denominator

        velocity_df = pd.DataFrame(
            {"article_id": pivot.index, "sales_velocity": slopes}
        )

        return velocity_df

    def _compute_affinity(
        self,
        transactions: pd.DataFrame,
        articles: pd.DataFrame,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Computes User-Item Affinity based on Department.
        Feature: user_dept_ratio (How much of user's history is in this candidate's department?)
        """
        logger.info("Computing User-Department Affinity...")

        # 1. Merge Department info to transactions
        # Ensure article_id types match
        if transactions["article_id"].dtype != object:
            transactions["article_id"] = transactions["article_id"].astype(str)
        if articles["article_id"].dtype != object:
            articles["article_id"] = articles["article_id"].astype(str)

        # Map article to department
        art_to_dept = dict(zip(articles["article_id"], articles["department_no"]))

        # Add department to transactions
        # Using map is faster than merge for single column
        transactions["department_no"] = transactions["article_id"].map(art_to_dept)

        # 2. Calculate User Purchase Counts per Department
        user_dept_counts = (
            transactions.groupby(["customer_id", "department_no"])
            .size()
            .reset_index(name="dept_count")
        )

        # 3. Calculate Total User Purchases
        user_total_counts = (
            transactions.groupby("customer_id").size().reset_index(name="total_count")
        )

        # 4. Merge to get Ratio
        affinity = pd.merge(user_dept_counts, user_total_counts, on="customer_id")
        affinity["user_dept_ratio"] = affinity["dept_count"] / affinity["total_count"]

        # 5. Prepare Candidate Data for Merge
        # We need to map candidates to departments
        candidates_temp = candidates[["customer_id", "article_id"]].copy()
        candidates_temp["department_no"] = candidates_temp["article_id"].map(
            art_to_dept
        )

        # 6. Merge Affinity into Candidates
        # Left join: if user has never bought from this dept, ratio is NaN (fill with 0)
        merged = pd.merge(
            candidates_temp,
            affinity[["customer_id", "department_no", "user_dept_ratio"]],
            on=["customer_id", "department_no"],
            how="left",
        )

        merged["user_dept_ratio"] = merged["user_dept_ratio"].fillna(0.0)

        return merged["user_dept_ratio"]

    def _compute_graph_features(
        self,
        candidates: pd.DataFrame,
        transactions: pd.DataFrame,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
        user_map: Dict[str, int],
        item_map: Dict[int, str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes:
        1. graph_dot_product: User Emb . Candidate Item Emb
        2. last_item_graph_similarity: Cosine(Last Purchased Item Emb, Candidate Item Emb)
        """
        logger.info("Computing Graph Features (Dot Product & Last-Item Similarity)...")

        # Reverse item map for lookup: article_id (str) -> index (int)
        item_str_to_idx = {v: k for k, v in item_map.items()}

        # Convert embeddings to Tensor on CPU (or GPU if available and fits)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        u_emb_t = torch.tensor(user_emb, device=device, dtype=torch.float32)
        i_emb_t = torch.tensor(item_emb, device=device, dtype=torch.float32)

        # Normalize item embeddings for Cosine Similarity later
        i_emb_norm = F.normalize(i_emb_t, p=2, dim=1)

        # --- Prepare Indices ---
        # 1. Candidate Indices
        # Filter candidates where user or item is not in graph maps (cold start)
        # We will compute features for valid ones and fill 0 for others

        valid_mask = candidates["customer_id"].isin(user_map) & candidates[
            "article_id"
        ].isin(item_str_to_idx)

        # Create arrays for indexing
        # Initialize results
        n_candidates = len(candidates)
        dot_products = np.zeros(n_candidates, dtype=np.float32)
        last_item_sims = np.zeros(n_candidates, dtype=np.float32)

        if not valid_mask.any():
            return dot_products, last_item_sims

        valid_indices = np.where(valid_mask)[0]
        valid_candidates = candidates.iloc[valid_indices]

        # Map IDs to indices
        u_indices = (
            valid_candidates["customer_id"].map(user_map).values.astype(np.int64)
        )
        i_indices = (
            valid_candidates["article_id"].map(item_str_to_idx).values.astype(np.int64)
        )

        u_idx_t = torch.tensor(u_indices, device=device)
        i_idx_t = torch.tensor(i_indices, device=device)

        # --- 1. Graph Dot Product ---
        # (Batch, Dim) * (Batch, Dim) -> (Batch,)
        # Gather embeddings
        batch_u = u_emb_t[u_idx_t]
        batch_i = i_emb_t[i_idx_t]

        scores = torch.sum(batch_u * batch_i, dim=1)
        dot_products[valid_indices] = scores.cpu().numpy()

        # --- 2. Last Item Graph Similarity ---
        # Find last item for each user
        logger.info("Identifying last purchased items...")
        last_purchases = (
            transactions.sort_values("t_dat").groupby("customer_id").tail(1)
        )
        last_item_map = dict(
            zip(last_purchases["customer_id"], last_purchases["article_id"])
        )

        # Map valid candidates' users to their last item
        # Note: Some users might be in graph but not in 'transactions' passed here if windows differ,
        # or have no last item in the graph's item set.

        # Get last item ID for the users in valid_candidates
        user_last_items = valid_candidates["customer_id"].map(last_item_map)

        # Check if last item exists in graph
        has_last_item_mask = user_last_items.isin(item_str_to_idx)

        # We only compute for those who have a valid last item
        sub_valid_indices = np.where(has_last_item_mask)[
            0
        ]  # Indices relative to valid_candidates

        if len(sub_valid_indices) > 0:
            # Indices in the original candidates dataframe
            original_indices = valid_indices[sub_valid_indices]

            # Get the item indices
            last_item_ids = user_last_items.iloc[sub_valid_indices]
            last_item_indices = last_item_ids.map(item_str_to_idx).values.astype(
                np.int64
            )

            # Candidate item indices for these rows
            cand_item_indices = i_indices[sub_valid_indices]

            # Tensors
            last_item_idx_t = torch.tensor(last_item_indices, device=device)
            cand_item_idx_t = torch.tensor(cand_item_indices, device=device)

            # Gather Normalized Embeddings
            batch_last = i_emb_norm[last_item_idx_t]
            batch_cand = i_emb_norm[cand_item_idx_t]

            # Cosine Similarity is dot product of normalized vectors
            sim_scores = torch.sum(batch_last * batch_cand, dim=1)

            last_item_sims[original_indices] = sim_scores.cpu().numpy()

        return dot_products, last_item_sims

    def generate_features(
        self,
        candidates: pd.DataFrame,
        transactions: pd.DataFrame,
        articles: pd.DataFrame,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
        user_map: Dict[str, int],
        item_map: Dict[int, str],
        load_cached: bool = True,
        suffix: str = "",
    ) -> pd.DataFrame:
        """
        Main method to generate and enrich features.

        Args:
            candidates: DataFrame with ['customer_id', 'article_id'].
            transactions: Historical transactions for feature calculation.
            articles: Metadata for articles.
            user_emb, item_emb: LightGCN embeddings.
            user_map, item_map: Mappings for graph.
            load_cached: Whether to load from cache.
            suffix: Identifier for cache file (e.g., 'train', 'val', 'test').

        Returns:
            DataFrame with original columns plus new features.
        """
        cache_file = f"features_{suffix}.parquet"

        if load_cached and self.cache.exists(cache_file):
            logger.info(f"Loading features from cache: {cache_file}")
            return self.cache.load_parquet(cache_file)

        logger.info(f"Generating features for {len(candidates)} candidates...")

        # Copy to avoid SettingWithCopy warnings on input
        df = candidates.copy()

        # Fill NaNs in score columns (Cite solution_lesson_node_00012)
        score_cols = ["cooc_score", "graph_score", "repurchase_score", "pop_score"]
        for col in score_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # 1. Sales Velocity
        velocity_df = self._compute_sales_velocity(transactions)
        # Merge
        df = pd.merge(df, velocity_df, on="article_id", how="left")
        df["sales_velocity"] = df["sales_velocity"].fillna(0.0)

        # 2. User-Item Affinity (Department Ratio)
        df["user_dept_ratio"] = self._compute_affinity(transactions, articles, df)

        # 3. Graph Features
        dot_prod, last_sim = self._compute_graph_features(
            df, transactions, user_emb, item_emb, user_map, item_map
        )
        df["graph_dot_product"] = dot_prod
        df["last_item_graph_similarity"] = last_sim

        # Optimize memory
        df = reduce_mem_usage(df)

        # Save
        self.cache.save_parquet(df, cache_file)

        return df
