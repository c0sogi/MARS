import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from library import config
from library import data_loader


class FeatureEngineer:
    def __init__(self):
        """
        Initializes the FeatureEngineer.
        Sets up the cache directory and prepares for lazy loading of the SBERT model.
        """
        self.cache_dir = config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.sbert_model = None

    def _get_sbert_model(self):
        """
        Lazy loader for the SentenceTransformer model to save resources if not used.
        """
        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(
                config.SBERT_MODEL_NAME, device=config.DEVICE
            )
        return self.sbert_model

    def generate_metadata_features(self, df, split_name="train", load_cached_data=True):
        """
        Generates full-spectrum metadata features including raw magnitudes,
        engineered ratios, and arcsinh-transformed versions.

        Args:
            df (pd.DataFrame): Input dataframe.
            split_name (str): Name of the split (train/val/test) for caching.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            pd.DataFrame: Dataframe with original and new features.
        """
        cache_file = os.path.join(self.cache_dir, f"metadata_{split_name}.parquet")

        if load_cached_data and os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        # 1. Select Base Numerical Columns (fill NaNs with 0 for safety)
        base_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

        # Create a copy to avoid SettingWithCopy warnings
        meta_df = df[base_cols].fillna(0).copy()

        # 2. Engineer Ratios
        epsilon = 1e-6

        # Upvote Ratio
        # Derived from: Plus = Up + Down, Minus = Up - Down
        # Up = (Plus + Minus) / 2
        up_votes = (
            meta_df["requester_upvotes_plus_downvotes_at_request"]
            + meta_df["requester_upvotes_minus_downvotes_at_request"]
        ) / 2
        meta_df["upvote_ratio"] = up_votes / (
            meta_df["requester_upvotes_plus_downvotes_at_request"] + epsilon
        )

        # RAOP Activity Ratios
        meta_df["raop_post_ratio"] = meta_df[
            "requester_number_of_posts_on_raop_at_request"
        ] / (meta_df["requester_number_of_posts_at_request"] + epsilon)
        meta_df["raop_comment_ratio"] = meta_df[
            "requester_number_of_comments_in_raop_at_request"
        ] / (meta_df["requester_number_of_comments_at_request"] + epsilon)

        # 3. Text Meta-Features
        meta_df["title_len_chars"] = (
            df["request_title"].fillna("").astype(str).apply(len)
        )
        meta_df["body_len_chars"] = (
            df["request_text_edit_aware"].fillna("").astype(str).apply(len)
        )

        # 4. Arcsinh Transformations
        # Apply to magnitude columns and length features
        cols_to_transform = base_cols + ["title_len_chars", "body_len_chars"]
        for col in cols_to_transform:
            meta_df[f"{col}_arcsinh"] = np.arcsinh(meta_df[col])

        # Save to cache
        meta_df.to_parquet(cache_file)
        return meta_df

    def compute_sbert_embeddings(self, df, split_name="train", load_cached_data=True):
        """
        Computes SBERT embeddings for:
        1. Request content (Title + Body).
        2. User history centroid (Average of subreddit embeddings).

        Args:
            df (pd.DataFrame): Input dataframe.
            split_name (str): Name of split for caching.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            tuple: (request_embeddings, history_centroids) as numpy arrays.
        """
        cache_file = os.path.join(self.cache_dir, f"sbert_{split_name}.npz")

        if load_cached_data and os.path.exists(cache_file):
            data = np.load(cache_file)
            return data["request_embeddings"], data["history_centroids"]

        model = self._get_sbert_model()

        # 1. Request Embeddings
        # Concatenate title and body
        texts = (
            df["request_title"].fillna("")
            + " "
            + df["request_text_edit_aware"].fillna("")
        ).tolist()
        request_embeddings = model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        # 2. History Centroids
        # Identify all unique subreddits in this dataset to batch encode efficiently
        all_subreddits = set()
        # df["requester_subreddits_at_request"] is expected to be a list of strings
        for sub_list in df["requester_subreddits_at_request"]:
            if isinstance(sub_list, list):
                all_subreddits.update(sub_list)

        unique_subs = list(all_subreddits)

        # Encode unique subreddits
        if unique_subs:
            sub_embeddings = model.encode(
                unique_subs,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            sub_map = {sub: emb for sub, emb in zip(unique_subs, sub_embeddings)}
            embedding_dim = sub_embeddings.shape[1]
        else:
            sub_map = {}
            embedding_dim = request_embeddings.shape[1]

        # Compute centroids per user
        history_centroids = []
        for sub_list in df["requester_subreddits_at_request"]:
            if isinstance(sub_list, list) and len(sub_list) > 0:
                # Gather embeddings for this user's subreddits
                user_sub_embs = [sub_map[s] for s in sub_list if s in sub_map]
                if user_sub_embs:
                    centroid = np.mean(user_sub_embs, axis=0)
                else:
                    centroid = np.zeros(embedding_dim)
            else:
                centroid = np.zeros(embedding_dim)
            history_centroids.append(centroid)

        history_centroids = np.array(history_centroids)

        # Save to cache
        np.savez(
            cache_file,
            request_embeddings=request_embeddings,
            history_centroids=history_centroids,
        )
        return request_embeddings, history_centroids

    def get_tfidf_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates Dual-Lexical TF-IDF features for Title and Body.
        Fits on Train, transforms Val and Test.

        Args:
            train_df, val_df, test_df: Dataframes for each split.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            tuple: (train_title, train_body, val_title, val_body, test_title, test_body)
                   as dense numpy arrays.
        """
        cache_file = os.path.join(self.cache_dir, "tfidf_features.npz")

        if load_cached_data and os.path.exists(cache_file):
            data = np.load(cache_file)
            return (
                data["train_title"],
                data["train_body"],
                data["val_title"],
                data["val_body"],
                data["test_title"],
                data["test_body"],
            )

        # 1. Title TF-IDF
        tfidf_title = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES, stop_words="english"
        )
        train_title = tfidf_title.fit_transform(
            train_df["request_title"].fillna("")
        ).toarray()
        val_title = tfidf_title.transform(val_df["request_title"].fillna("")).toarray()
        test_title = tfidf_title.transform(
            test_df["request_title"].fillna("")
        ).toarray()

        # 2. Body TF-IDF
        tfidf_body = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES, stop_words="english"
        )
        train_body = tfidf_body.fit_transform(
            train_df["request_text_edit_aware"].fillna("")
        ).toarray()
        val_body = tfidf_body.transform(
            val_df["request_text_edit_aware"].fillna("")
        ).toarray()
        test_body = tfidf_body.transform(
            test_df["request_text_edit_aware"].fillna("")
        ).toarray()

        # Save to cache
        np.savez(
            cache_file,
            train_title=train_title,
            train_body=train_body,
            val_title=val_title,
            val_body=val_body,
            test_title=test_title,
            test_body=test_body,
        )

        return train_title, train_body, val_title, val_body, test_title, test_body
