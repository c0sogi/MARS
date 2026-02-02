import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import torch

from library.config import Config
from library.utils import Timer, set_seed


class FeatureFactory:
    """
    Factory class for generating multi-modal features with strict caching.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        Config.ensure_directories()
        set_seed(Config.RANDOM_SEED)

    def _get_cache_paths(self, feature_name, extension):
        """Helper to generate cache paths for train, val, test."""
        return (
            os.path.join(self.cache_dir, f"X_train_{feature_name}.{extension}"),
            os.path.join(self.cache_dir, f"X_val_{feature_name}.{extension}"),
            os.path.join(self.cache_dir, f"X_test_{feature_name}.{extension}"),
        )

    def _save_cache(self, paths, data_tuple, is_sparse=False):
        """Helper to save data to cache."""
        train_path, val_path, test_path = paths
        X_train, X_val, X_test = data_tuple

        if is_sparse:
            sp.save_npz(train_path, X_train)
            sp.save_npz(val_path, X_val)
            sp.save_npz(test_path, X_test)
        else:
            np.save(train_path, X_train)
            np.save(val_path, X_val)
            np.save(test_path, X_test)

    def _load_cache(self, paths, is_sparse=False):
        """Helper to load data from cache."""
        train_path, val_path, test_path = paths
        if not (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            return None

        if is_sparse:
            return (
                sp.load_npz(train_path),
                sp.load_npz(val_path),
                sp.load_npz(test_path),
            )
        else:
            return (
                np.load(train_path),
                np.load(val_path),
                np.load(test_path),
            )

    def get_lexical_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates TF-IDF features from combined text (Title + Body).
        Returns sparse matrices.
        """
        paths = self._get_cache_paths("lexical", "npz")

        if load_cached_data:
            cached = self._load_cache(paths, is_sparse=True)
            if cached:
                print("[FeatureFactory] Loaded Lexical features from cache.")
                return cached

        with Timer("Lexical Feature Generation"):
            print("[FeatureFactory] Generating Lexical features...")

            # Text is already combined in 'text_combined' by data_processing
            train_text = train_df["text_combined"].fillna("").astype(str)
            val_text = val_df["text_combined"].fillna("").astype(str)
            test_text = test_df["text_combined"].fillna("").astype(str)

            vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)
            X_train = vectorizer.fit_transform(train_text)
            X_val = vectorizer.transform(val_text)
            X_test = vectorizer.transform(test_text)

            self._save_cache(paths, (X_train, X_val, X_test), is_sparse=True)

        return X_train, X_val, X_test

    def get_behavioral_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates TF-IDF features from subreddit history (Bag-of-Concepts).
        Returns sparse matrices.
        """
        paths = self._get_cache_paths("community", "npz")

        if load_cached_data:
            cached = self._load_cache(paths, is_sparse=True)
            if cached:
                print("[FeatureFactory] Loaded Behavioral features from cache.")
                return cached

        with Timer("Behavioral Feature Generation"):
            print("[FeatureFactory] Generating Behavioral features...")

            def process_subreddits(series):
                # Convert list of subreddits to space-separated string
                return series.apply(
                    lambda x: " ".join(x) if isinstance(x, list) else ""
                )

            train_subs = process_subreddits(train_df[Config.SUBREDDIT_COL])
            val_subs = process_subreddits(val_df[Config.SUBREDDIT_COL])
            test_subs = process_subreddits(test_df[Config.SUBREDDIT_COL])

            # Use TF-IDF but limit features to top K subreddits
            vectorizer = TfidfVectorizer(
                max_features=Config.TOP_K_SUBREDDITS,
                stop_words="english",
                binary=True,  # Presence matters more than frequency in history often
            )

            X_train = vectorizer.fit_transform(train_subs)
            X_val = vectorizer.transform(val_subs)
            X_test = vectorizer.transform(test_subs)

            self._save_cache(paths, (X_train, X_val, X_test), is_sparse=True)

        return X_train, X_val, X_test

    def get_semantic_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates dense embeddings using SentenceTransformers.
        Returns numpy arrays.
        """
        paths = self._get_cache_paths("semantic", "npy")

        if load_cached_data:
            cached = self._load_cache(paths, is_sparse=False)
            if cached:
                print("[FeatureFactory] Loaded Semantic features from cache.")
                return cached

        with Timer("Semantic Feature Generation"):
            print("[FeatureFactory] Generating Semantic features (Embeddings)...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME, device=device)

            train_text = train_df["text_combined"].fillna("").astype(str).tolist()
            val_text = val_df["text_combined"].fillna("").astype(str).tolist()
            test_text = test_df["text_combined"].fillna("").astype(str).tolist()

            # Encode
            X_train = model.encode(
                train_text,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            X_val = model.encode(
                val_text, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            X_test = model.encode(
                test_text, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )

            self._save_cache(paths, (X_train, X_val, X_test), is_sparse=False)

        return X_train, X_val, X_test

    def get_metadata_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Extracts pre-processed dense metadata columns.
        Returns numpy arrays.
        """
        paths = self._get_cache_paths("metadata", "npy")

        if load_cached_data:
            cached = self._load_cache(paths, is_sparse=False)
            if cached:
                print("[FeatureFactory] Loaded Metadata features from cache.")
                return cached

        with Timer("Metadata Feature Extraction"):
            print("[FeatureFactory] Extracting Metadata features...")

            # Columns are already scaled in data_processing.py
            # We just need to ensure we select the right columns in order
            cols = [c for c in Config.METADATA_DENSE_COLS if c in train_df.columns]

            X_train = train_df[cols].to_numpy(dtype=np.float32)
            X_val = val_df[cols].to_numpy(dtype=np.float32)
            X_test = test_df[cols].to_numpy(dtype=np.float32)

            self._save_cache(paths, (X_train, X_val, X_test), is_sparse=False)

        return X_train, X_val, X_test

    def get_latent_interaction_features(
        self,
        X_train_lex,
        X_train_beh,
        X_train_meta,
        X_val_lex,
        X_val_beh,
        X_val_meta,
        X_test_lex,
        X_test_beh,
        X_test_meta,
        load_cached_data=True,
    ):
        """
        Generates Latent Interaction features by applying SVD to Lexical and Behavioral
        matrices and concatenating them with Metadata.
        Returns numpy arrays.
        """
        paths = self._get_cache_paths("interaction", "npy")

        if load_cached_data:
            cached = self._load_cache(paths, is_sparse=False)
            if cached:
                print("[FeatureFactory] Loaded Latent Interaction features from cache.")
                return cached

        with Timer("Latent Interaction Feature Generation"):
            print(
                "[FeatureFactory] Computing SVD and concatenating interaction features..."
            )

            # 1. SVD on Lexical Features
            svd_text = TruncatedSVD(
                n_components=Config.SVD_COMPONENTS_TEXT, random_state=Config.RANDOM_SEED
            )
            X_train_svd_text = svd_text.fit_transform(X_train_lex)
            X_val_svd_text = svd_text.transform(X_val_lex)
            X_test_svd_text = svd_text.transform(X_test_lex)

            # 2. SVD on Behavioral Features
            # Note: X_train_beh might have fewer columns than n_components if vocabulary is small
            n_comp_hist = min(Config.SVD_COMPONENTS_HISTORY, X_train_beh.shape[1] - 1)
            svd_hist = TruncatedSVD(
                n_components=n_comp_hist, random_state=Config.RANDOM_SEED
            )
            X_train_svd_hist = svd_hist.fit_transform(X_train_beh)
            X_val_svd_hist = svd_hist.transform(X_val_beh)
            X_test_svd_hist = svd_hist.transform(X_test_beh)

            # 3. Concatenate: SVD_Text + SVD_History + Metadata
            X_train = np.hstack([X_train_svd_text, X_train_svd_hist, X_train_meta])
            X_val = np.hstack([X_val_svd_text, X_val_svd_hist, X_val_meta])
            X_test = np.hstack([X_test_svd_text, X_test_svd_hist, X_test_meta])

            self._save_cache(paths, (X_train, X_val, X_test), is_sparse=False)

        return X_train, X_val, X_test
