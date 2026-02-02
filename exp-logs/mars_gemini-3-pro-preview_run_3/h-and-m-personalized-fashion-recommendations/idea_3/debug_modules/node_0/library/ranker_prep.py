import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from library import config
from library import utils
from library import data_loader
from library import image_processor
from library import sparse_engine


class RankerDatasetBuilder:
    """
    Constructs datasets for the LightGBM ranker by integrating:
    1. Sparse Retrieval Candidates (Stage 1)
    2. Visual Similarity Features
    3. Metadata Features
    4. Ground Truth Labels
    """

    def __init__(self):
        self.working_dir = config.WORKING_DIR
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Cache paths
        self.train_path = config.CACHE_RANKER_TRAIN
        self.val_path = config.CACHE_RANKER_VAL
        self.test_path = self.working_dir / "ranker_test_set.parquet"

    def _get_time_split(
        self, df: pd.DataFrame, days: int = 7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits a transaction dataframe into History (pre-cutoff) and Target (post-cutoff).
        Cutoff is max_date - days.
        """
        if "t_dat" not in df.columns:
            df["t_dat"] = pd.to_datetime(df["t_dat"])

        max_date = df["t_dat"].max()
        cutoff_date = max_date - pd.Timedelta(days=days)

        history = df[df["t_dat"] < cutoff_date].copy()
        target = df[df["t_dat"] >= cutoff_date].copy()

        return history, target

    def _compute_visual_similarity(
        self,
        candidates_df: pd.DataFrame,
        history_df: pd.DataFrame,
        article_embeddings: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Computes cosine similarity between user history centroid and candidate items.
        Returns an array of similarity scores aligned with candidates_df.
        """
        print("Computing visual similarity features...")

        # 1. Prepare Embeddings Matrix
        # Convert dict to matrix for fast lookup
        # We need a consistent mapping.
        # Let's rely on the article_ids present in the data.

        # Filter history to relevant users
        target_users = candidates_df["customer_id"].unique()
        hist_subset = history_df[history_df["customer_id"].isin(target_users)].copy()

        # 2. Compute User Centroids
        # We'll do this by aggregating embeddings.
        # To be efficient, we map article_id to embedding vector in the dataframe.

        # It's faster to do a matrix operation if possible, but users have different history lengths.
        # Let's use pandas groupby with a custom aggregation or apply.
        # Optimization: Pre-compute embedding matrix for all articles in history

        # Get all unique articles in history
        unique_hist_arts = hist_subset["article_id"].unique()

        # Create a temporary lookup for history items
        # Handle missing embeddings by using zero vector (though ImageEmbedder handles imputation)
        emb_dim = config.IMAGE_EMBEDDING_DIM
        default_emb = np.zeros(emb_dim, dtype=np.float32)

        # Build matrix for history items
        # Map article_id -> index
        hist_art_to_idx = {aid: i for i, aid in enumerate(unique_hist_arts)}
        hist_emb_matrix = np.zeros((len(unique_hist_arts), emb_dim), dtype=np.float32)

        for aid, idx in hist_art_to_idx.items():
            hist_emb_matrix[idx] = article_embeddings.get(aid, default_emb)

        # Map history dataframe to these indices
        hist_subset["emb_idx"] = hist_subset["article_id"].map(hist_art_to_idx)

        # Now we want to sum embeddings per user.
        # We can use scipy.sparse to sum efficiently: (n_users, n_hist_items) @ (n_hist_items, emb_dim)
        user_to_idx = {uid: i for i, uid in enumerate(target_users)}
        n_users = len(target_users)

        # Filter out rows where article mapping failed (shouldn't happen with correct logic)
        hist_subset = hist_subset.dropna(subset=["emb_idx"])

        rows = hist_subset["customer_id"].map(user_to_idx).values
        cols = hist_subset["emb_idx"].values
        data = np.ones(
            len(rows), dtype=np.float32
        )  # Simple average, or use time decay here?
        # Let's use simple average for visual preference stability

        from scipy import sparse

        user_item_mat = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_users, len(unique_hist_arts))
        )

        # Sum embeddings
        user_sums = user_item_mat.dot(hist_emb_matrix)

        # Count items per user for mean
        user_counts = np.array(user_item_mat.sum(axis=1)).flatten()
        user_counts[user_counts == 0] = 1  # Avoid div by zero

        user_centroids = user_sums / user_counts[:, None]

        # Normalize centroids for cosine similarity
        norms = np.linalg.norm(user_centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1
        user_centroids_norm = user_centroids / norms

        # 3. Compute Candidate Embeddings
        # Map candidates to embeddings
        cand_arts = candidates_df["article_id"].values
        cand_users = candidates_df["customer_id"].values

        # We need to compute dot product: Centroid(u) . Item(i)
        # Construct arrays aligned with candidates_df

        # Array of user centroids aligned with rows
        # Map user_id -> row index in user_centroids
        u_indices = np.array([user_to_idx.get(u, -1) for u in cand_users])

        # Filter valid users (some candidates might be for users with no history?
        # SparseRetriever handles this, but let's be safe)
        valid_mask = u_indices != -1

        # Get candidate item embeddings
        # We can't vectorize dict lookup easily, but we can build a matrix for candidate items
        unique_cand_arts = np.unique(cand_arts)
        cand_art_to_idx = {aid: i for i, aid in enumerate(unique_cand_arts)}
        cand_emb_matrix = np.zeros((len(unique_cand_arts), emb_dim), dtype=np.float32)

        for aid, idx in cand_art_to_idx.items():
            cand_emb_matrix[idx] = article_embeddings.get(aid, default_emb)

        # Normalize candidate embeddings
        c_norms = np.linalg.norm(cand_emb_matrix, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1
        cand_emb_matrix_norm = cand_emb_matrix / c_norms

        # Gather aligned arrays
        # Aligned User Centroids
        aligned_centroids = np.zeros((len(candidates_df), emb_dim), dtype=np.float32)
        aligned_centroids[valid_mask] = user_centroids_norm[u_indices[valid_mask]]

        # Aligned Item Embeddings
        c_indices = np.array([cand_art_to_idx[aid] for aid in cand_arts])
        aligned_items = cand_emb_matrix_norm[c_indices]

        # Dot product (Cosine Similarity since normalized)
        # sum(A * B, axis=1)
        similarities = np.sum(aligned_centroids * aligned_items, axis=1)

        return similarities

    def _add_metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges customer and article metadata.
        """
        print("Merging metadata features...")

        # Load Raw Metadata
        articles = data_loader.load_articles(load_cached_data=True)
        customers = data_loader.load_customers(load_cached_data=True)

        # Select features
        # Articles
        art_cols = [
            "article_id",
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
            "department_no",
            "index_group_no",
            "section_no",
            "garment_group_no",
        ]
        # Customers
        cust_cols = [
            "customer_id",
            "FN",
            "Active",
            "club_member_status",
            "fashion_news_frequency",
            "age",
        ]

        # Merge
        df = df.merge(articles[art_cols], on="article_id", how="left")
        df = df.merge(customers[cust_cols], on="customer_id", how="left")

        # Preprocessing
        # Fill NaNs
        df["age"] = df["age"].fillna(df["age"].median())
        df["FN"] = df["FN"].fillna(0)
        df["Active"] = df["Active"].fillna(0)

        # Label Encode Categoricals for LightGBM
        # Note: LightGBM can handle categories, but we need them as ints.
        # We use a simple hash or mapping if cardinality is low.
        # For this exercise, we assume the raw IDs (like product_type_no) are sufficient integers.
        # For string columns in customers:
        for col in ["club_member_status", "fashion_news_frequency"]:
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        return df

    def build_ranker_train_set(self, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Generates the training dataset for the ranker.
        Uses the last 7 days of the training metadata as the target.
        """
        if load_cached_data and self.train_path.exists():
            print(f"Loading Ranker Train Set from {self.train_path}")
            return pd.read_parquet(self.train_path)

        with utils.Timer("Build Ranker Train Set"):
            # 1. Load & Split Data
            train_meta = data_loader.load_transactions(split="train")
            history_df, target_df = self._get_time_split(train_meta, days=7)

            # 2. Fit Retriever on History
            retriever = sparse_engine.SparseGraphRetriever()
            # We force re-fit to ensure T is built only on history (no leakage)
            retriever.fit(history_df, load_cached_data=False)

            # 3. Generate Candidates
            # Only for users active in the target week
            target_users = target_df["customer_id"].unique()
            print(f"Generating candidates for {len(target_users)} training users...")

            candidates = retriever.predict(target_users, user_history_df=history_df)

            # 4. Create Labels
            print("Creating labels...")
            # Create a set of (user, item) tuples present in target
            # Optimization: Use a merged flag
            target_df["purchased"] = 1
            # We only need user, item, purchased
            truth = target_df[
                ["customer_id", "article_id", "purchased"]
            ].drop_duplicates()

            candidates = candidates.merge(
                truth, on=["customer_id", "article_id"], how="left"
            )
            candidates["label"] = candidates["purchased"].fillna(0).astype(int)
            candidates.drop(columns=["purchased"], inplace=True)

            # 5. Visual Features
            # Load embeddings
            embedder = image_processor.ImageEmbedder()
            articles_df = data_loader.load_articles()
            embeddings = embedder.get_embeddings(articles_df)

            sim_scores = self._compute_visual_similarity(
                candidates, history_df, embeddings
            )
            candidates["visual_similarity"] = sim_scores

            # 6. Metadata Features
            candidates = self._add_metadata_features(candidates)

            # 7. Save
            print(f"Saving to {self.train_path}...")
            candidates.to_parquet(self.train_path, index=False)

            # Cleanup
            del history_df, target_df, train_meta, embeddings
            gc.collect()

            return candidates

    def build_ranker_val_set(self, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Generates the validation dataset for the ranker.
        Uses the validation metadata (hold-out users).
        """
        if load_cached_data and self.val_path.exists():
            print(f"Loading Ranker Val Set from {self.val_path}")
            return pd.read_parquet(self.val_path)

        with utils.Timer("Build Ranker Val Set"):
            # 1. Load Data
            val_meta = data_loader.load_transactions(split="val")
            history_df, target_df = self._get_time_split(val_meta, days=7)

            # 2. Load Retriever
            # We use the retriever fitted on TRAIN history (from build_ranker_train_set)
            # This tests generalization to new users using the learned Item-Item graph.
            # NOTE: This assumes build_ranker_train_set was run recently and cache exists.
            # If not, we should fit it.
            retriever = sparse_engine.SparseGraphRetriever()
            if not retriever._check_cache_exists():
                print("Retriever cache missing. Fitting on Train History first...")
                train_meta = data_loader.load_transactions(split="train")
                h_train, _ = self._get_time_split(train_meta, days=7)
                retriever.fit(h_train, load_cached_data=False)
                del train_meta, h_train
                gc.collect()
            else:
                retriever._load_cache()

            # 3. Generate Candidates
            target_users = target_df["customer_id"].unique()
            print(f"Generating candidates for {len(target_users)} validation users...")

            # Important: Pass val history so retriever knows what these users bought
            candidates = retriever.predict(target_users, user_history_df=history_df)

            # 4. Labels
            target_df["purchased"] = 1
            truth = target_df[
                ["customer_id", "article_id", "purchased"]
            ].drop_duplicates()
            candidates = candidates.merge(
                truth, on=["customer_id", "article_id"], how="left"
            )
            candidates["label"] = candidates["purchased"].fillna(0).astype(int)
            candidates.drop(columns=["purchased"], inplace=True)

            # 5. Visual Features
            embedder = image_processor.ImageEmbedder()
            articles_df = data_loader.load_articles()
            embeddings = embedder.get_embeddings(articles_df)

            sim_scores = self._compute_visual_similarity(
                candidates, history_df, embeddings
            )
            candidates["visual_similarity"] = sim_scores

            # 6. Metadata
            candidates = self._add_metadata_features(candidates)

            # 7. Save
            print(f"Saving to {self.val_path}...")
            candidates.to_parquet(self.val_path, index=False)

            return candidates

    def build_inference_set(self, load_cached_data: bool = True) -> pd.DataFrame:
        """
        Generates the dataset for final inference (Test Users).
        Uses ALL available history (Train + Val).
        """
        if load_cached_data and self.test_path.exists():
            print(f"Loading Ranker Test Set from {self.test_path}")
            return pd.read_parquet(self.test_path)

        with utils.Timer("Build Ranker Inference Set"):
            # 1. Load All History
            train_meta = data_loader.load_transactions(split="train")
            val_meta = data_loader.load_transactions(split="val")
            full_history = pd.concat([train_meta, val_meta], ignore_index=True)

            # 2. Fit Retriever on Full History
            print("Fitting retriever on full combined history...")
            retriever = sparse_engine.SparseGraphRetriever()
            retriever.fit(full_history, load_cached_data=False)

            # 3. Load Test Users
            sample_sub = data_loader.load_sample_submission()
            test_users = sample_sub["customer_id"].unique()

            # 4. Generate Candidates
            # Note: Test users might be in history (returning users) or not (cold).
            # We pass full_history so the retriever can look up their past interactions.
            candidates = retriever.predict(test_users, user_history_df=full_history)

            # 5. Visual Features
            embedder = image_processor.ImageEmbedder()
            articles_df = data_loader.load_articles()
            embeddings = embedder.get_embeddings(articles_df)

            sim_scores = self._compute_visual_similarity(
                candidates, full_history, embeddings
            )
            candidates["visual_similarity"] = sim_scores

            # 6. Metadata
            candidates = self._add_metadata_features(candidates)

            # 7. Save
            print(f"Saving to {self.test_path}...")
            candidates.to_parquet(self.test_path, index=False)

            # Cleanup
            del train_meta, val_meta, full_history, embeddings
            gc.collect()

            return candidates
