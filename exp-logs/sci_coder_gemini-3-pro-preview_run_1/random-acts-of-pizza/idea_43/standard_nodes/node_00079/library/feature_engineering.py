import os
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from typing import Dict, List, Tuple, Optional

from library.config import Config
from library.utils import set_seed


class FeatureProcessor:
    """
    Handles feature engineering for the Hybrid Ensemble solution.
    Generates SBERT embeddings, User History Statistics, Metadata, and Top-K indicators.
    """

    def __init__(self):
        self.sbert_model = None
        self.scaler = StandardScaler()
        self.top_k_subreddits = []
        self.vader = None

        # Ensure NLTK data is available
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

    def _load_sbert(self):
        """Lazy loader for SBERT model."""
        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(
                Config.SBERT_MODEL_NAME, device=Config.DEVICE
            )

    def _load_vader(self):
        """Lazy loader for VADER."""
        if self.vader is None:
            self.vader = SentimentIntensityAnalyzer()

    def compute_sbert_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Generates dense vector embeddings for a list of texts using SBERT.
        """
        self._load_sbert()
        # Handle empty or non-string inputs gracefully
        cleaned_texts = [str(t) if pd.notna(t) else "" for t in texts]
        embeddings = self.sbert_model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings

    def compute_history_stats(
        self,
        df: pd.DataFrame,
        title_embeddings: np.ndarray,
        body_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes User History Centroids, Global Consistency Scalars, and padded History Sequences.

        Returns:
            centroids (np.ndarray): (N, 384) Mean embedding of user history.
            consistency (np.ndarray): (N, 2) Cosine similarity [Title-Centroid, Body-Centroid].
            sequences (np.ndarray): (N, Max_Len, 384) Padded sequence of history embeddings.
        """
        self._load_sbert()

        # 1. Collect all unique subreddits to batch encode
        all_subreddits = set()
        for sub_list in df["requester_subreddits_at_request"]:
            if isinstance(sub_list, list):
                all_subreddits.update(sub_list)
            elif isinstance(sub_list, np.ndarray):
                all_subreddits.update(sub_list.tolist())

        unique_subs = list(all_subreddits)
        if not unique_subs:
            # Handle edge case where no subreddits exist
            N = len(df)
            dim = Config.EMBEDDING_DIM
            return np.zeros((N, dim)), np.zeros((N, 2)), np.zeros((N, 1, dim))

        # 2. Encode unique subreddits
        sub_embeddings_map = {}
        sub_embeddings = self.sbert_model.encode(
            unique_subs, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        for sub, emb in zip(unique_subs, sub_embeddings):
            sub_embeddings_map[sub] = emb

        # 3. Aggregate per user
        centroids = []
        sequences = []
        max_len = 0

        # First pass to find max length (optional, or fix a reasonable max)
        # We'll use a dynamic max length based on the batch

        temp_sequences = []

        for sub_list in df["requester_subreddits_at_request"]:
            user_embs = []
            if isinstance(sub_list, (list, np.ndarray)):
                for sub in sub_list:
                    if sub in sub_embeddings_map:
                        user_embs.append(sub_embeddings_map[sub])

            if user_embs:
                user_embs_arr = np.array(user_embs)
                centroid = np.mean(user_embs_arr, axis=0)
                temp_sequences.append(user_embs_arr)
                max_len = max(max_len, len(user_embs))
            else:
                # No history
                centroid = np.zeros(Config.EMBEDDING_DIM)
                temp_sequences.append(np.zeros((0, Config.EMBEDDING_DIM)))

            centroids.append(centroid)

        centroids = np.array(centroids)

        # Pad sequences
        # Limit max_len to avoid excessive memory usage if someone has 1000 subs
        LIMIT_LEN = 50
        final_max_len = min(max_len, LIMIT_LEN)
        if final_max_len == 0:
            final_max_len = 1  # Avoid zero dim

        padded_sequences = np.zeros(
            (len(df), final_max_len, Config.EMBEDDING_DIM), dtype=np.float32
        )

        for i, seq in enumerate(temp_sequences):
            if len(seq) > 0:
                length = min(len(seq), final_max_len)
                padded_sequences[i, :length, :] = seq[:length]

        # 4. Compute Consistency Scalars
        # Cosine similarity between Request Title and History Centroid
        # Cosine similarity between Request Body and History Centroid

        # Reshape for sklearn cosine_similarity (1, D) vs (1, D) is inefficient in loop
        # Do dot product manually for speed: (A . B) / (|A| |B|)

        def cosine_sim_batch(A, B):
            # A, B are (N, D)
            dot = np.sum(A * B, axis=1)
            norm_a = np.linalg.norm(A, axis=1)
            norm_b = np.linalg.norm(B, axis=1)
            # Avoid divide by zero
            denom = norm_a * norm_b
            return np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)

        sim_title = cosine_sim_batch(title_embeddings, centroids)
        sim_body = cosine_sim_batch(body_embeddings, centroids)

        consistency = np.stack([sim_title, sim_body], axis=1)

        return (
            centroids.astype(np.float32),
            consistency.astype(np.float32),
            padded_sequences,
        )

    def extract_metadata(self, df: pd.DataFrame, is_train: bool = False) -> np.ndarray:
        """
        Extracts numerical metadata, applies transformations, and scales features.
        """
        self._load_vader()

        # 1. Base Numerical Features
        num_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
            "requester_number_of_subreddits_at_request",
        ]

        # Fill NaNs
        X_num = df[num_cols].fillna(0).values.astype(np.float32)

        # Apply Arcsinh transformation (handles 0 and negative values better than log)
        X_num = np.arcsinh(X_num)

        # 2. Engineered Ratios
        # Upvote Ratio: (Up - Down) / (Up + Down) -> approximate via columns
        # We have (U-D) and (U+D).
        # Ratio = (U-D) / (U+D). If U+D is 0, ratio is 0.
        u_minus_d = df["requester_upvotes_minus_downvotes_at_request"].fillna(0).values
        u_plus_d = df["requester_upvotes_plus_downvotes_at_request"].fillna(0).values

        ratio = np.divide(
            u_minus_d,
            u_plus_d,
            out=np.zeros_like(u_minus_d, dtype=float),
            where=u_plus_d != 0,
        )

        # 3. Text Meta-Features
        # Length of title, length of body, caps ratio
        titles = df[Config.TEXT_COL_TITLE].fillna("").astype(str)
        bodies = df[Config.TEXT_COL_BODY].fillna("").astype(str)

        title_len = titles.apply(len).values
        body_len = bodies.apply(len).values

        def get_caps_ratio(s):
            if len(s) == 0:
                return 0.0
            return sum(1 for c in s if c.isupper()) / len(s)

        title_caps = titles.apply(get_caps_ratio).values

        # 4. Sentiment Analysis
        # VADER compound score for title and body
        def get_sentiment(s):
            return self.vader.polarity_scores(s)["compound"]

        title_sent = titles.apply(get_sentiment).values
        body_sent = bodies.apply(get_sentiment).values

        # Concatenate all raw features
        # Shape: (N, 7 + 1 + 3 + 2) = (N, 13)
        X_combined = np.column_stack(
            [X_num, ratio, title_len, body_len, title_caps, title_sent, body_sent]
        )

        # Scale
        if is_train:
            X_scaled = self.scaler.fit_transform(X_combined)
        else:
            X_scaled = self.scaler.transform(X_combined)

        return X_scaled.astype(np.float32)

    def identify_top_k_subreddits(
        self, df: pd.DataFrame, is_train: bool = False
    ) -> np.ndarray:
        """
        Creates binary indicator vectors for the top K most frequent subreddits.
        """
        if is_train:
            # Flatten all subreddits
            all_subs = []
            for sub_list in df["requester_subreddits_at_request"]:
                if isinstance(sub_list, list):
                    all_subs.extend(sub_list)
                elif isinstance(sub_list, np.ndarray):
                    all_subs.extend(sub_list.tolist())

            # Find top K
            counts = Counter(all_subs)
            self.top_k_subreddits = [
                sub for sub, _ in counts.most_common(Config.TOP_K_SUBREDDITS)
            ]

        # Create binary matrix
        K = len(self.top_k_subreddits)
        N = len(df)
        X_topk = np.zeros((N, K), dtype=np.float32)

        top_k_set = {sub: i for i, sub in enumerate(self.top_k_subreddits)}

        for idx, sub_list in enumerate(df["requester_subreddits_at_request"]):
            if isinstance(sub_list, (list, np.ndarray)):
                for sub in sub_list:
                    if sub in top_k_set:
                        col_idx = top_k_set[sub]
                        X_topk[idx, col_idx] = 1.0

        return X_topk

    def process_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        load_cached_data: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Main pipeline to process all data splits.
        Checks cache, computes features if missing, and saves to cache.
        """
        set_seed()

        # Define cache file paths
        cache_files = {
            "train": os.path.join(Config.WORKING_DIR, "features_train.npz"),
            "val": os.path.join(Config.WORKING_DIR, "features_val.npz"),
            "test": os.path.join(Config.WORKING_DIR, "features_test.npz"),
        }

        # Check if all cache files exist
        all_exist = all(os.path.exists(f) for f in cache_files.values())

        if load_cached_data and all_exist:
            print("Loading features from cache...")
            data = {}
            for split, path in cache_files.items():
                loaded = np.load(path)
                data[split] = {k: loaded[k] for k in loaded.files}
            return data

        print("Computing features from scratch...")

        # 1. Top-K Subreddits (Fit on Train)
        print("Processing Top-K Subreddits...")
        X_topk_train = self.identify_top_k_subreddits(train_df, is_train=True)
        X_topk_val = self.identify_top_k_subreddits(val_df, is_train=False)
        X_topk_test = self.identify_top_k_subreddits(test_df, is_train=False)

        # 2. Metadata (Fit Scaler on Train)
        print("Processing Metadata...")
        X_meta_train = self.extract_metadata(train_df, is_train=True)
        X_meta_val = self.extract_metadata(val_df, is_train=False)
        X_meta_test = self.extract_metadata(test_df, is_train=False)

        # 3. Text & History Embeddings (SBERT)
        # Helper to process one dataframe
        def process_split(df):
            # Title & Body
            emb_title = self.compute_sbert_embeddings(
                df[Config.TEXT_COL_TITLE].tolist()
            )
            emb_body = self.compute_sbert_embeddings(df[Config.TEXT_COL_BODY].tolist())

            # History
            centroids, consistency, sequences = self.compute_history_stats(
                df, emb_title, emb_body
            )

            return {
                "emb_title": emb_title,
                "emb_body": emb_body,
                "history_centroids": centroids,
                "consistency": consistency,
                "history_sequences": sequences,
            }

        print("Processing SBERT Embeddings (Train)...")
        sbert_train = process_split(train_df)
        print("Processing SBERT Embeddings (Val)...")
        sbert_val = process_split(val_df)
        print("Processing SBERT Embeddings (Test)...")
        sbert_test = process_split(test_df)

        # 4. Assemble and Save
        results = {}

        def save_split(name, X_meta, X_topk, sbert_dict, df_target=None):
            split_data = {"X_meta": X_meta, "X_topk": X_topk, **sbert_dict}

            # Add target if available
            if df_target is not None and Config.TARGET_COL in df_target.columns:
                split_data["y"] = df_target[Config.TARGET_COL].astype(int).values

            # Add IDs for reference
            if Config.ID_COL in df_target.columns:
                split_data["ids"] = df_target[Config.ID_COL].values.astype(str)

            # Save to disk
            np.savez_compressed(cache_files[name], **split_data)
            results[name] = split_data

        save_split("train", X_meta_train, X_topk_train, sbert_train, train_df)
        save_split("val", X_meta_val, X_topk_val, sbert_val, val_df)
        save_split("test", X_meta_test, X_topk_test, sbert_test, test_df)

        print("Feature processing complete and cached.")
        return results
