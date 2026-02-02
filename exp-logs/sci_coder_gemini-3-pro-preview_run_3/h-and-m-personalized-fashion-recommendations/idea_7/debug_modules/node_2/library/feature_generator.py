import os
import gc
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import normalize
from library.config import Config
from library.data_utils import get_id_maps, load_dataset, get_sliding_windows
from library.retrieval_engine import DualGraphRetriever
from library.visual_encoder import ImageEmbedder


class RankerFeatureFactory:
    """
    Generates dense feature vectors for the Ranking Stage (LightGBM).
    Merges retrieval candidates with metadata, visual consistency signals,
    and global popularity metrics.
    """

    def __init__(self):
        # Load ID mappings
        self.cust_to_idx, self.art_to_idx, self.cust_map, self.art_map = get_id_maps()

        # Load Embeddings
        # We use the ImageEmbedder class to handle loading/generation logic
        embedder = ImageEmbedder()
        self.article_embeddings = embedder.generate_embeddings(load_cached_data=True)

        # Normalize embeddings for cosine similarity calculations
        self.article_embeddings = normalize(self.article_embeddings, axis=1, norm="l2")

        # Initialize Retriever
        self.retriever = DualGraphRetriever()

        # Load Raw Metadata for Feature Merging
        _, _, _, self.articles_df, self.customers_df = load_dataset()

        # Pre-process Customers DF for merging
        # Select relevant columns and fill NaNs
        self.cust_feats = self.customers_df[
            ["customer_id", "age", "fashion_news_frequency", "club_member_status"]
        ].copy()
        self.cust_feats["age"] = self.cust_feats["age"].fillna(-1)
        self.cust_feats["fashion_news_frequency"] = self.cust_feats[
            "fashion_news_frequency"
        ].fillna("NONE")
        self.cust_feats["club_member_status"] = self.cust_feats[
            "club_member_status"
        ].fillna("NONE")

        # Encode Categorical Customer Features
        for col in ["fashion_news_frequency", "club_member_status"]:
            self.cust_feats[col] = self.cust_feats[col].astype("category").cat.codes

        # Pre-process Articles DF for merging
        # Select relevant columns
        self.art_cols = [
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
        self.art_feats = self.articles_df[self.art_cols].copy()
        # Fill NaNs if any (articles usually clean)
        self.art_feats = self.art_feats.fillna(-1)

    def _compute_user_centroids(self, history_df):
        """
        Computes the visual centroid of each user's history.
        Centroid = Normalized(Weighted Sum of Item Embeddings)

        Returns:
            np.array: Matrix of shape (num_users, embedding_dim)
        """
        print("Computing user visual centroids...")

        # 1. Get Sparse User History Matrix (Users x Items)
        # Values are time-decayed weights
        # We use the retriever's utility for this
        user_matrix = self.retriever.get_user_history_matrix(history_df)

        # 2. Compute Weighted Sum of Embeddings
        # U (N_users x N_items) @ E (N_items x Dim) -> C (N_users x Dim)
        # This effectively calculates the weighted average vector of items bought by the user
        centroids = user_matrix.dot(self.article_embeddings)

        # 3. Normalize Centroids
        # Avoid division by zero for users with no history
        centroids = normalize(centroids, axis=1, norm="l2")

        return centroids

    def _add_features(self, candidates_df, history_df):
        """
        Enriches the candidates DataFrame with features.
        """
        print("Enriching candidates with features...")

        # --- 1. Global Popularity (in history period) ---
        pop_counts = history_df["article_id"].value_counts().reset_index()
        pop_counts.columns = ["article_id", "popularity_count"]

        # Merge popularity
        candidates_df = candidates_df.merge(pop_counts, on="article_id", how="left")
        candidates_df["popularity_count"] = candidates_df["popularity_count"].fillna(0)

        # --- 2. Visual Consistency ---
        # Compute centroids based on the specific history provided
        centroids = self._compute_user_centroids(history_df)

        # Map IDs to indices for fast lookup
        # Candidates df has raw IDs.
        # We need to map them to the indices used in centroids (global customer idx)
        # and article_embeddings (global article idx)

        # Filter candidates to ensure IDs exist in maps (safety check)
        valid_mask = (candidates_df["customer_id"].isin(self.cust_to_idx)) & (
            candidates_df["article_id"].isin(self.art_to_idx)
        )

        # We will compute consistency only for valid rows, others get 0
        valid_candidates = candidates_df[valid_mask].copy()

        if len(valid_candidates) > 0:
            cust_indices = valid_candidates["customer_id"].map(self.cust_to_idx).values
            art_indices = valid_candidates["article_id"].map(self.art_to_idx).values

            # Gather vectors
            user_vecs = centroids[cust_indices]
            item_vecs = self.article_embeddings[art_indices]

            # Compute Cosine Similarity (Dot product of normalized vectors)
            # Row-wise dot product
            consistency = np.sum(user_vecs * item_vecs, axis=1)

            # Assign back
            candidates_df.loc[valid_mask, "visual_consistency"] = consistency

        candidates_df["visual_consistency"] = candidates_df[
            "visual_consistency"
        ].fillna(0.0)

        # --- 3. Metadata Merging ---
        # Merge Customer Features
        candidates_df = candidates_df.merge(
            self.cust_feats, on="customer_id", how="left"
        )

        # Merge Article Features
        candidates_df = candidates_df.merge(self.art_feats, on="article_id", how="left")

        return candidates_df

    def create_train_dataset(self, load_cached_data=True):
        """
        Generates the training dataset for the ranker using a sliding window strategy.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        if load_cached_data and Config.CACHE_RANKER_TRAIN.exists():
            print(
                f"Loading cached Ranker Train set from {Config.CACHE_RANKER_TRAIN}..."
            )
            return pd.read_parquet(Config.CACHE_RANKER_TRAIN)

        print("Generating Ranker Train set from scratch...")

        # Load transactions
        train_df, _, _, _, _ = load_dataset()

        # Define Windows
        max_date = train_df["t_dat"].max()
        windows = get_sliding_windows(
            max_date, num_windows=5
        )  # 5 windows as per common practice

        all_features = []

        for i, (hist_start, hist_end, target_start, target_end) in enumerate(windows):
            print(f"\nProcessing Window {i+1}/{len(windows)}")
            print(f"History: {hist_start.date()} to {hist_end.date()}")
            print(f"Target:  {target_start.date()} to {target_end.date()}")

            # 1. Split Data
            hist_df = train_df[
                (train_df["t_dat"] >= hist_start) & (train_df["t_dat"] <= hist_end)
            ]
            target_df = train_df[
                (train_df["t_dat"] >= target_start) & (train_df["t_dat"] <= target_end)
            ]

            # Identify target users (active in history AND target ideally, but we predict for all active in history)
            # To train the ranker effectively, we focus on users who actually made purchases in the target week
            # so we have positive labels.
            target_users = target_df["customer_id"].unique()

            # Downsample for training speed if needed?
            # With 12 CPUs and 220GB RAM, we can handle a lot, but let's be safe.
            # Let's take all target users.

            # 2. Generate Candidates (Retrieval)
            # This uses the history window to predict
            candidates = self.retriever.generate_candidates(
                hist_df, target_users, load_cached_graphs=False
            )

            # 3. Create Labels
            # Create a set of (user, item) tuples present in target_df
            target_df["purchased"] = 1
            # We aggregate to handle multiple purchases of same item?
            # Binary classification usually sufficient for ranking (1 if bought).
            target_pairs = target_df[["customer_id", "article_id"]].drop_duplicates()
            target_pairs["label"] = 1

            # Merge labels
            candidates = candidates.merge(
                target_pairs, on=["customer_id", "article_id"], how="left"
            )
            candidates["label"] = candidates["label"].fillna(0).astype(int)

            # 4. Add Features
            candidates = self._add_features(candidates, hist_df)

            all_features.append(candidates)

            # Cleanup
            del hist_df, target_df, candidates, target_pairs
            gc.collect()

        # Concatenate all windows
        full_train_df = pd.concat(all_features, ignore_index=True)

        # Save
        print(f"Saving Ranker Train set to {Config.CACHE_RANKER_TRAIN}...")
        full_train_df.to_parquet(Config.CACHE_RANKER_TRAIN, index=False)

        return full_train_df

    def create_validation_dataset(self, load_cached_data=True):
        """
        Generates a validation set (single window, usually the last one before test).
        Actually, in the sliding window approach, the last window IS the validation if we hold it out.
        Or we can use the provided val_df (metadata split).

        Let's use the provided val_df as the target, and the end of train_df as history.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        if load_cached_data and Config.CACHE_RANKER_VAL.exists():
            print(f"Loading cached Ranker Val set from {Config.CACHE_RANKER_VAL}...")
            return pd.read_parquet(Config.CACHE_RANKER_VAL)

        print("Generating Ranker Validation set from scratch...")

        train_df, val_df, _, _, _ = load_dataset()

        # History is the last N weeks of training data
        max_train_date = train_df["t_dat"].max()
        hist_start = max_train_date - timedelta(weeks=Config.RETRIEVAL_HISTORY_WEEKS)
        hist_df = train_df[train_df["t_dat"] > hist_start]

        # Target is val_df
        target_users = val_df["customer_id"].unique()

        # Generate Candidates
        candidates = self.retriever.generate_candidates(
            hist_df, target_users, load_cached_graphs=True
        )

        # Labels
        val_pairs = val_df[["customer_id", "article_id"]].drop_duplicates()
        val_pairs["label"] = 1

        candidates = candidates.merge(
            val_pairs, on=["customer_id", "article_id"], how="left"
        )
        candidates["label"] = candidates["label"].fillna(0).astype(int)

        # Features
        candidates = self._add_features(candidates, hist_df)

        # Save
        print(f"Saving Ranker Val set to {Config.CACHE_RANKER_VAL}...")
        candidates.to_parquet(Config.CACHE_RANKER_VAL, index=False)

        return candidates

    def create_inference_dataset(self, load_cached_data=True):
        """
        Generates the dataset for final inference (Test Set).
        """
        # We don't cache inference dataset usually as it's generated on the fly for submission,
        # but caching helps if we crash.
        cache_path = Config.WORKING_DIR / "ranker_inference.parquet"

        if load_cached_data and cache_path.exists():
            print(f"Loading cached Ranker Inference set from {cache_path}...")
            return pd.read_parquet(cache_path)

        print("Generating Ranker Inference set from scratch...")

        train_df, val_df, test_df, _, _ = load_dataset()

        # Combine train and val for maximum history
        full_history = pd.concat([train_df, val_df], ignore_index=True)

        # Filter strictly by recency for the graph construction
        max_date = full_history["t_dat"].max()
        hist_start = max_date - timedelta(weeks=Config.RETRIEVAL_HISTORY_WEEKS)
        hist_df = full_history[full_history["t_dat"] > hist_start]

        # Target Users: All users in sample submission
        target_users = test_df["customer_id"].unique()

        # Generate Candidates
        candidates = self.retriever.generate_candidates(
            hist_df, target_users, load_cached_graphs=True
        )

        # Features (No labels for inference)
        candidates = self._add_features(candidates, hist_df)

        # Save
        print(f"Saving Ranker Inference set to {cache_path}...")
        candidates.to_parquet(cache_path, index=False)

        return candidates
