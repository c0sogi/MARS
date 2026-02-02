import pandas as pd
import numpy as np
import torch
import gc
import os
from pathlib import Path
from tqdm import tqdm
from scipy import sparse
from library import config
from library import utils
from library import heuristics


class FeatureBuilder:
    """
    Constructs features for the ranking stage (Stage 2).
    Combines Behavioral, Affinity, Trend, and Contextual signals.
    """

    def __init__(self):
        self.device = config.DEVICE

    def compute_features(
        self,
        candidates_df,
        history_df,
        articles_df,
        customers_df,
        seq_data=None,
        user_embeddings=None,
        item_embeddings=None,
        cooc_matrix=None,
        load_cached_data=False,
        cache_name="features_ranker.parquet",
    ):
        """
        Main method to compute features for the provided candidates.

        Args:
            candidates_df (pd.DataFrame): (customer_id, article_id) pairs.
            history_df (pd.DataFrame): Transaction history.
            articles_df (pd.DataFrame): Article metadata.
            customers_df (pd.DataFrame): Customer metadata.
            seq_data (dict): Data dict from sequence preprocessing (maps, etc.).
            user_embeddings (np.array): User state vectors from Seq Model.
            item_embeddings (np.array): Item embeddings from Seq Model.
            cooc_matrix (heuristics.CooccurrenceMatrix): Fitted co-occurrence model.
            load_cached_data (bool): Whether to load from cache.
            cache_name (str): Filename for caching.

        Returns:
            pd.DataFrame: Feature matrix ready for LightGBM.
        """
        # Ensure working directory exists
        os.makedirs(config.WORKING_DIR, exist_ok=True)
        cache_path = config.WORKING_DIR / cache_name

        # 1. Caching Logic
        if load_cached_data and cache_path.exists():
            print(f"Loading cached features from {cache_path}...")
            return pd.read_parquet(cache_path)

        print("Computing features from scratch...")

        # Ensure base dataframes are optimized
        candidates_df = utils.reduce_mem_usage(candidates_df.copy())

        # 2. Metadata Features (Merge)
        print("Adding Metadata Features...")
        features_df = self._add_metadata(candidates_df, articles_df, customers_df)

        # 3. Behavioral: Repurchase Stats (Source C)
        print("Adding Repurchase Stats...")
        features_df = self._add_repurchase_stats(features_df, history_df)

        # 4. Trend: Sales Velocity (Source D)
        print("Adding Trend Features...")
        features_df = self._add_trend_features(features_df, history_df)

        # 5. Affinity Features
        print("Adding Affinity Features...")
        features_df = self._add_affinity_features(features_df, history_df, articles_df)

        # 6. Sequential Model Features (Source B)
        if (
            seq_data is not None
            and user_embeddings is not None
            and item_embeddings is not None
        ):
            print("Adding Sequential Model Features...")
            features_df = self._add_sequential_features(
                features_df, seq_data, user_embeddings, item_embeddings
            )
        else:
            print("Skipping Sequential Features (inputs missing).")

        # 7. Co-occurrence Features (Source A)
        if cooc_matrix is not None:
            print("Adding Co-occurrence Features...")
            features_df = self._add_cooc_features(features_df, history_df, cooc_matrix)
        else:
            print("Skipping Co-occurrence Features (matrix missing).")

        # 8. Final Cleanup and Save
        features_df = utils.reduce_mem_usage(features_df)

        print(f"Saving features to {cache_path}...")
        features_df.to_parquet(cache_path, index=False)

        return features_df

    def _add_metadata(self, df, articles_df, customers_df):
        """Merges article and customer metadata."""
        # Merge Articles
        # Select relevant columns to save memory
        art_cols = ["article_id"] + config.ARTICLE_CATEGORICAL_FEATURES
        # Ensure article_id match types
        df = df.merge(articles_df[art_cols], on="article_id", how="left")

        # Merge Customers
        cust_cols = ["customer_id"] + config.CUSTOMER_FEATURES
        df = df.merge(customers_df[cust_cols], on="customer_id", how="left")

        # Encode Categoricals
        # FN and Active are usually NaN or 1.
        df["FN"] = df["FN"].fillna(0).astype(int)
        df["Active"] = df["Active"].fillna(0).astype(int)

        # Club member status - Label Encode if string
        if df["club_member_status"].dtype == "object":
            df["club_member_status"] = (
                df["club_member_status"].astype("category").cat.codes
            )

        # Fashion news frequency
        if df["fashion_news_frequency"].dtype == "object":
            df["fashion_news_frequency"] = (
                df["fashion_news_frequency"].astype("category").cat.codes
            )

        return df

    def _add_repurchase_stats(self, df, history_df):
        """Adds repurchase frequency and recency."""
        # Calculate last purchase date and count for every user-item pair in history
        if not np.issubdtype(history_df["t_dat"].dtype, np.datetime64):
            history_df["t_dat"] = pd.to_datetime(history_df["t_dat"])

        max_date = history_df["t_dat"].max()

        stats = (
            history_df.groupby(["customer_id", "article_id"])
            .agg(
                purchase_count=("article_id", "count"),
                last_purchase_date=("t_dat", "max"),
            )
            .reset_index()
        )

        stats["days_since_last_purchase"] = (
            max_date - stats["last_purchase_date"]
        ).dt.days

        # Merge
        df = df.merge(
            stats[
                [
                    "customer_id",
                    "article_id",
                    "purchase_count",
                    "days_since_last_purchase",
                ]
            ],
            on=["customer_id", "article_id"],
            how="left",
        )

        # Fill NaNs (Items never bought by user)
        df["purchase_count"] = df["purchase_count"].fillna(0)
        df["days_since_last_purchase"] = df["days_since_last_purchase"].fillna(
            999
        )  # Large number for never

        return df

    def _add_trend_features(self, df, history_df):
        """Adds sales velocity (slope) and recent popularity."""
        # Filter to last 14 days
        if not np.issubdtype(history_df["t_dat"].dtype, np.datetime64):
            history_df["t_dat"] = pd.to_datetime(history_df["t_dat"])

        max_date = history_df["t_dat"].max()
        start_date = max_date - pd.Timedelta(days=14)
        recent = history_df[history_df["t_dat"] > start_date].copy()

        # Let's use the simple difference method: (Week 2 - Week 1)
        split_date = max_date - pd.Timedelta(days=7)

        week1 = recent[recent["t_dat"] <= split_date].groupby("article_id").size()
        week2 = recent[recent["t_dat"] > split_date].groupby("article_id").size()

        trend_df = pd.DataFrame(index=recent["article_id"].unique())
        trend_df["count_w1"] = week1
        trend_df["count_w2"] = week2
        trend_df = trend_df.fillna(0)

        # Trend score: normalized difference
        trend_df["sales_trend"] = (trend_df["count_w2"] - trend_df["count_w1"]) / (
            trend_df["count_w1"] + 1
        )
        trend_df["recent_volume"] = trend_df["count_w2"]

        # Merge
        df = df.merge(
            trend_df[["sales_trend", "recent_volume"]],
            left_on="article_id",
            right_index=True,
            how="left",
        )
        df["sales_trend"] = df["sales_trend"].fillna(0)
        df["recent_volume"] = df["recent_volume"].fillna(0)

        return df

    def _add_affinity_features(self, df, history_df, articles_df):
        """
        Adds user affinity to article categories (e.g. department).
        Feature: User's purchase rate in the candidate's department.
        """
        # Join history with articles to get categories
        hist_merged = history_df[["customer_id", "article_id"]].merge(
            articles_df[["article_id", "department_no", "garment_group_no"]],
            on="article_id",
            how="left",
        )

        # Helper to calculate affinity
        def calc_affinity(col_name):
            # Count user purchases in each category
            user_cat_counts = (
                hist_merged.groupby(["customer_id", col_name])
                .size()
                .reset_index(name="cat_count")
            )

            # Total user purchases
            user_total = (
                hist_merged.groupby("customer_id")
                .size()
                .reset_index(name="total_count")
            )

            # Merge
            affinity = user_cat_counts.merge(user_total, on="customer_id")
            affinity[f"affinity_{col_name}"] = (
                affinity["cat_count"] / affinity["total_count"]
            )

            return affinity[["customer_id", col_name, f"affinity_{col_name}"]]

        # 1. Department Affinity
        dept_aff = calc_affinity("department_no")
        df = df.merge(dept_aff, on=["customer_id", "department_no"], how="left")
        df["affinity_department_no"] = df["affinity_department_no"].fillna(0)

        # 2. Garment Group Affinity
        garm_aff = calc_affinity("garment_group_no")
        df = df.merge(garm_aff, on=["customer_id", "garment_group_no"], how="left")
        df["affinity_garment_group_no"] = df["affinity_garment_group_no"].fillna(0)

        return df

    def _add_sequential_features(self, df, seq_data, user_embeddings, item_embeddings):
        """
        Adds cosine similarity between User State and Item Embedding.
        Also adds Last-Item Similarity.
        Cite solution_lesson_node_00021: Recovering Sequential Signals via Last-Item Similarity.
        """
        # Maps
        user_ids_in_emb = seq_data[
            "customer_ids"
        ]  # Array of customer_ids corresponding to user_embeddings rows
        article_map = seq_data["article_map"]  # article_id -> int index
        sequences = seq_data["sequences"].numpy()  # (N_users, L)

        # Create User Map: customer_id -> row index in user_embeddings
        user_map = {uid: i for i, uid in enumerate(user_ids_in_emb)}

        # Map indices
        df["user_emb_idx"] = df["customer_id"].map(user_map)
        df["item_emb_idx"] = df["article_id"].map(article_map)

        # Filter valid rows
        valid_mask = df["user_emb_idx"].notna() & df["item_emb_idx"].notna()

        # Initialize score columns
        df["seq_similarity"] = 0.0
        df["last_item_similarity"] = 0.0

        if valid_mask.sum() > 0:
            valid_indices = df[valid_mask].index

            u_indices = df.loc[valid_indices, "user_emb_idx"].astype(int).values
            i_indices = df.loc[valid_indices, "item_emb_idx"].astype(int).values

            # Gather embeddings
            u_vecs = user_embeddings[u_indices]
            i_vecs = item_embeddings[i_indices]

            # 1. User Mean Similarity
            scores = np.sum(u_vecs * i_vecs, axis=1)
            df.loc[valid_indices, "seq_similarity"] = scores

            # 2. Last Item Similarity
            # Extract last item index for each user (vectorized)
            # Default to 0 (padding)
            last_item_indices_all = np.zeros(len(user_ids_in_emb), dtype=int)

            # Iterate columns to find last non-zero
            for col in range(sequences.shape[1]):
                vals = sequences[:, col]
                mask = vals > 0
                last_item_indices_all[mask] = vals[mask]

            # Gather for specific rows
            u_last_items = last_item_indices_all[u_indices]

            # Get embeddings of last items
            last_item_vecs = item_embeddings[u_last_items]

            # Dot product with candidate item
            last_scores = np.sum(last_item_vecs * i_vecs, axis=1)

            # If last item was 0 (no history), score is 0
            has_history = u_last_items > 0
            df.loc[valid_indices, "last_item_similarity"] = last_scores * has_history

        # Drop temp columns
        df = df.drop(columns=["user_emb_idx", "item_emb_idx"])

        return df

    def _add_cooc_features(self, df, history_df, cooc_model):
        """
        Adds co-occurrence score: Sum of similarities between candidate and user's history.
        """
        # Get maps from model
        article_map = cooc_model.article_map
        matrix = cooc_model.matrix  # Item-Item similarity

        unique_users = df["customer_id"].unique()

        # Filter history to these users and items in map
        relevant_hist = history_df[
            (history_df["customer_id"].isin(unique_users))
            & (history_df["article_id"].isin(article_map))
        ].copy()

        # Local user map for batch processing
        user_list = list(unique_users)
        local_user_map = {uid: i for i, uid in enumerate(user_list)}

        relevant_hist["local_u_idx"] = relevant_hist["customer_id"].map(local_user_map)
        relevant_hist["item_idx"] = relevant_hist["article_id"].map(article_map)

        # Build User-Item binary history matrix for these users
        # Shape: (N_unique_users, N_items)
        U_sparse = sparse.csr_matrix(
            (
                np.ones(len(relevant_hist)),
                (relevant_hist["local_u_idx"], relevant_hist["item_idx"]),
            ),
            shape=(len(user_list), len(article_map)),
        )

        # Strategy: Process in batches of users to keep memory low
        batch_size = 1000
        df["cooc_score"] = 0.0

        # Pre-map df to local indices
        df["local_u_idx"] = df["customer_id"].map(local_user_map)
        df["item_idx"] = df["article_id"].map(article_map)

        # We iterate users in batches
        for start in tqdm(range(0, len(user_list), batch_size), desc="Cooc Feature"):
            end = min(start + batch_size, len(user_list))

            # Slice user history
            u_batch = U_sparse[start:end]

            # Compute scores for all items for these users
            # (Batch, Items) * (Items, Items) -> (Batch, Items)
            scores_batch = u_batch.dot(matrix)

            # Identify rows in df belonging to this batch
            mask = (df["local_u_idx"] >= start) & (df["local_u_idx"] < end)
            if not mask.any():
                continue

            batch_df_indices = df[mask].index

            # Relative user index in batch
            rel_u_idx = (
                (df.loc[batch_df_indices, "local_u_idx"] - start).astype(int).values
            )
            i_idx = df.loc[batch_df_indices, "item_idx"].fillna(-1).astype(int).values

            # Handle items not in map (-1)
            valid_items = i_idx >= 0

            # Extract
            if sparse.issparse(scores_batch):
                scores_dense = scores_batch.toarray()
                extracted_scores = scores_dense[
                    rel_u_idx[valid_items], i_idx[valid_items]
                ]
            else:
                extracted_scores = scores_batch[
                    rel_u_idx[valid_items], i_idx[valid_items]
                ]

            # Assign
            final_values = np.zeros(len(batch_df_indices))
            final_values[valid_items] = extracted_scores

            df.loc[batch_df_indices, "cooc_score"] = final_values

        # Cleanup
        df = df.drop(columns=["local_u_idx", "item_idx"])
        return df
