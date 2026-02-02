import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from library.config import Config
from library.data_loader import get_feature_intersection


class FeatureProcessor:
    """
    Handles feature engineering for the Hybrid Ensemble (RF + MLP).
    Generates semantic embeddings, interaction features, and community profiles.
    """

    def __init__(self):
        self.sbert_model = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.imputer = None
        self.top_k_subreddits = None
        self.safe_numeric_cols = None

    def _init_sbert(self):
        if self.sbert_model is None:
            print(f"Loading SBERT model: {Config.SBERT_MODEL_NAME}...")
            self.sbert_model = SentenceTransformer(
                Config.SBERT_MODEL_NAME, device=Config.DEVICE
            )

    def _compute_sbert_embeddings(self, texts, batch_size=32):
        self._init_sbert()
        # Sort by length to minimize padding in batches (handled internally by ST usually, but good practice)
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def _compute_history_embeddings(
        self, df, unique_subreddits_map, title_embeddings, body_embeddings
    ):
        """
        Computes aggregated embeddings for user history:
        1. Centroid: Mean of all subreddit embeddings.
        2. Title-Attended: Weighted mean based on similarity to Title.
        3. Body-Attended: Weighted mean based on similarity to Body.
        """
        hist_centroids = []
        hist_title_attn = []
        hist_body_attn = []

        # Pre-allocate zero vector for empty histories
        emb_dim = list(unique_subreddits_map.values())[0].shape[0]
        zero_vec = np.zeros(emb_dim, dtype=np.float32)

        # Iterate through dataframe
        # Note: This loop can be slow, but history lengths are generally small.
        # Vectorization is hard due to variable length lists.

        subreddits_col = df["requester_subreddits_at_request"].tolist()

        for i, subs in enumerate(subreddits_col):
            if not subs:
                hist_centroids.append(zero_vec)
                hist_title_attn.append(zero_vec)
                hist_body_attn.append(zero_vec)
                continue

            # Retrieve embeddings for this user's subreddits
            # Filter out subs that might not be in the map (rare edge case if map built on all data)
            sub_embs = np.array(
                [unique_subreddits_map[s] for s in subs if s in unique_subreddits_map]
            )

            if len(sub_embs) == 0:
                hist_centroids.append(zero_vec)
                hist_title_attn.append(zero_vec)
                hist_body_attn.append(zero_vec)
                continue

            # 1. Centroid
            centroid = np.mean(sub_embs, axis=0)
            hist_centroids.append(centroid)

            # 2. Attention (Softmax of dot product)
            # Query: Title/Body [Dim], Key: Sub_Embs [N, Dim]
            # Scores: [N]

            # Title Attention
            t_query = title_embeddings[i]
            t_scores = np.dot(sub_embs, t_query)
            t_weights = np.exp(t_scores) / (np.sum(np.exp(t_scores)) + 1e-9)
            t_attn = np.sum(sub_embs * t_weights[:, None], axis=0)
            hist_title_attn.append(t_attn)

            # Body Attention
            b_query = body_embeddings[i]
            b_scores = np.dot(sub_embs, b_query)
            b_weights = np.exp(b_scores) / (np.sum(np.exp(b_scores)) + 1e-9)
            b_attn = np.sum(sub_embs * b_weights[:, None], axis=0)
            hist_body_attn.append(b_attn)

        return (
            np.array(hist_centroids),
            np.array(hist_title_attn),
            np.array(hist_body_attn),
        )

    def _get_top_k_subreddits(self, train_df, k=Config.TOP_K_SUBREDDITS):
        all_subs = [
            s
            for sublist in train_df["requester_subreddits_at_request"]
            for s in sublist
        ]
        counts = pd.Series(all_subs).value_counts()
        return counts.head(k).index.tolist()

    def _create_top_k_features(self, df):
        # Create binary matrix for top K subreddits
        # Returns numpy array [N, K]
        matrix = np.zeros((len(df), len(self.top_k_subreddits)), dtype=np.float32)

        # Map sub to index
        sub_to_idx = {sub: i for i, sub in enumerate(self.top_k_subreddits)}

        for row_idx, subs in enumerate(df["requester_subreddits_at_request"]):
            for s in subs:
                if s in sub_to_idx:
                    matrix[row_idx, sub_to_idx[s]] = 1.0
        return matrix

    def _extract_interaction_features(self, df, consistency_title, consistency_body):
        """
        Creates explicit interaction features for RF:
        I1 = Topic_Consistency * log(1 + Account_Age)
        I2 = Narrative_Consistency * Upvote_Ratio
        """
        # 1. Account Age
        age = df["requester_account_age_in_days_at_request"].fillna(0).values
        log_age = np.log1p(age)

        # 2. Upvote Ratio
        # U + D
        total = df["requester_upvotes_plus_downvotes_at_request"].fillna(0).values
        # U - D
        diff = df["requester_upvotes_minus_downvotes_at_request"].fillna(0).values

        # U = (Total + Diff) / 2
        upvotes = (total + diff) / 2

        # Avoid div by zero
        ratio = np.divide(
            upvotes, total, out=np.full_like(upvotes, 0.5), where=total != 0
        )

        # Interactions
        i1 = consistency_title * log_age
        i2 = consistency_body * ratio

        return np.stack([i1, i2], axis=1)

    def process_data(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main pipeline to process data.
        Returns a dictionary containing processed features for RF and MLP.
        """
        cache_file = os.path.join(Config.WORKING_DIR, "features_all.npz")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading features from {cache_file}...")
            try:
                data = np.load(cache_file, allow_pickle=True)
                return {
                    k: (
                        data[k].item()
                        if data[k].dtype == object and data[k].ndim == 0
                        else data[k]
                    )
                    for k in data.files
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print("Starting feature engineering...")

        # 1. Identify Safe Numeric Features
        self.safe_numeric_cols = get_feature_intersection(train_df, test_df)
        # Filter for numeric only
        self.safe_numeric_cols = [
            c
            for c in self.safe_numeric_cols
            if pd.api.types.is_numeric_dtype(train_df[c])
        ]

        # 2. Text Processing (SBERT)
        print("Encoding text with SBERT...")
        # Concatenate all unique subreddits to encode once
        all_subs = set()
        for df in [train_df, val_df, test_df]:
            for subs in df["requester_subreddits_at_request"]:
                all_subs.update(subs)

        unique_subs_list = list(all_subs)
        # Encode unique subreddits
        if unique_subs_list:
            sub_embeddings_arr = self._compute_sbert_embeddings(
                unique_subs_list, batch_size=256
            )
            sub_map = {
                sub: emb for sub, emb in zip(unique_subs_list, sub_embeddings_arr)
            }
        else:
            sub_map = {}
            # Fallback dim
            dummy_emb = self._compute_sbert_embeddings(["dummy"])[0]
            sub_map = {"dummy": dummy_emb}  # Should not be hit if data is clean

        # Process each split
        processed_splits = {}

        # Fit TF-IDF on Train
        print("Fitting TF-IDF...")
        train_text = (
            train_df["request_title"] + " " + train_df["request_text_edit_aware"]
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES, stop_words="english"
        )
        self.tfidf_vectorizer.fit(train_text)

        # Fit Scaler/Imputer on Train Numerics
        print("Fitting Scaler/Imputer...")
        train_nums = train_df[self.safe_numeric_cols].values
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # Impute then fit scaler
        train_nums_imputed = self.imputer.fit_transform(train_nums)
        # Arcsinh transform before scaling for MLP
        train_nums_arcsinh = np.arcsinh(train_nums_imputed)
        self.scaler.fit(train_nums_arcsinh)

        # Identify Top K Subreddits
        self.top_k_subreddits = self._get_top_k_subreddits(train_df)

        for split_name, df in zip(
            ["train", "val", "test"], [train_df, val_df, test_df]
        ):
            print(f"Processing {split_name} split...")

            # A. SBERT Features
            title_embs = self._compute_sbert_embeddings(df["request_title"].tolist())
            body_embs = self._compute_sbert_embeddings(
                df["request_text_edit_aware"].tolist()
            )

            # History Embeddings
            hist_cent, hist_t_attn, hist_b_attn = self._compute_history_embeddings(
                df, sub_map, title_embs, body_embs
            )

            # B. Consistency Scalars (Dot Product)
            # Cosine sim is dot product since embeddings are normalized
            cons_title = np.sum(title_embs * hist_cent, axis=1)
            cons_body = np.sum(body_embs * hist_cent, axis=1)

            # C. Top-K Community Flags
            top_k_flags = self._create_top_k_features(df)

            # D. Numeric Metadata
            raw_nums = df[self.safe_numeric_cols].values
            # Impute
            nums_imputed = self.imputer.transform(raw_nums)
            # Arcsinh + Scale (For MLP)
            nums_scaled = self.scaler.transform(np.arcsinh(nums_imputed))

            # E. Interaction Features (For RF)
            interactions = self._extract_interaction_features(df, cons_title, cons_body)

            # F. TF-IDF (For RF)
            text_combined = df["request_title"] + " " + df["request_text_edit_aware"]
            tfidf_feats = self.tfidf_vectorizer.transform(text_combined)

            # --- Assemble RF Features ---
            # [TF-IDF (Sparse), Numeric (Imputed), Top-K, Interactions, Consistency]
            # Convert dense parts to sparse to stack with TF-IDF

            # Add consistency scalars to numeric block for RF
            rf_dense_block = np.column_stack(
                [nums_imputed, top_k_flags, interactions, cons_title, cons_body]
            )

            rf_features = sparse.hstack(
                [tfidf_feats, sparse.csr_matrix(rf_dense_block)]
            )

            # --- Assemble MLP Features ---
            # Branch 1: Semantic [Title, Body, Hist_Cent, Hist_T_Attn, Hist_B_Attn, Cons_Scalars]
            mlp_sem = np.column_stack(
                [
                    title_embs,
                    body_embs,
                    hist_cent,
                    hist_t_attn,
                    hist_b_attn,
                    cons_title[:, None],
                    cons_body[:, None],
                ]
            ).astype(np.float32)

            # Branch 2: Reliability (Scaled Numerics)
            mlp_rel = nums_scaled.astype(np.float32)

            # Branch 3: Community (Top-K + Scaled Numerics)
            mlp_comm = np.column_stack([top_k_flags, nums_scaled]).astype(np.float32)

            # Target
            y = None
            if "requester_received_pizza" in df.columns:
                y = df["requester_received_pizza"].astype(int).values

            processed_splits[f"{split_name}_rf"] = rf_features
            processed_splits[f"{split_name}_mlp_sem"] = mlp_sem
            processed_splits[f"{split_name}_mlp_rel"] = mlp_rel
            processed_splits[f"{split_name}_mlp_comm"] = mlp_comm
            if y is not None:
                processed_splits[f"{split_name}_y"] = y

        # Save to cache
        print(f"Saving features to {cache_file}...")
        np.savez_compressed(cache_file, **processed_splits)

        return processed_splits
