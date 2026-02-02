import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch

from library import config
from library import utils


class FeatureEngineer:
    """
    Manages feature extraction for the stacking ensemble.
    Generates Holistic, Lexical, Community, Semantic, and Metadata views.
    """

    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        utils.ensure_dir(self.cache_dir)

        # Initialize TF-IDF Vectorizers
        self.holistic_vectorizer = TfidfVectorizer(**config.TFIDF_PARAMS)
        self.lexical_vectorizer = TfidfVectorizer(**config.TFIDF_PARAMS)

        # Community vectorizer has a specific max_features limit
        community_params = config.TFIDF_PARAMS.copy()
        community_params["max_features"] = config.COMMUNITY_MAX_FEATURES
        self.community_vectorizer = TfidfVectorizer(**community_params)

        # Embedding model (lazy loaded)
        self.embedding_model = None

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_embedding_model(self):
        if self.embedding_model is None:
            print(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
            self.embedding_model = SentenceTransformer(
                config.EMBEDDING_MODEL, device=self._get_device()
            )

    def _process_subreddits_to_string(self, df):
        """Helper to ensure subreddits are space-separated strings."""

        # Note: data_processing.py might have already handled this for 'holistic_text',
        # but we need it isolated for the Community view.
        def to_string(x):
            if isinstance(x, (list, np.ndarray)):
                return " ".join([str(s) for s in x if s])
            return str(x) if x is not None else ""

        return df[config.SUBREDDIT_COL].apply(to_string)

    def _create_text_features(self, train_df, val_df, test_df):
        """Generates sparse TF-IDF features."""
        print("Generating Sparse Features (Holistic, Lexical, Community)...")

        # 1. Holistic View (Title + Body + History) - Column already exists
        train_holistic = train_df["holistic_text"].fillna("").astype(str)
        val_holistic = val_df["holistic_text"].fillna("").astype(str)
        test_holistic = test_df["holistic_text"].fillna("").astype(str)

        print("Fitting Holistic Vectorizer...")
        X_train_holistic = self.holistic_vectorizer.fit_transform(train_holistic)
        X_val_holistic = self.holistic_vectorizer.transform(val_holistic)
        X_test_holistic = self.holistic_vectorizer.transform(test_holistic)

        # 2. Lexical View (Title + Body)
        def get_lexical_text(df):
            return (
                df["request_title"].fillna("")
                + " "
                + df["request_text_edit_aware"].fillna("")
            ).astype(str)

        train_lexical = get_lexical_text(train_df)
        val_lexical = get_lexical_text(val_df)
        test_lexical = get_lexical_text(test_df)

        print("Fitting Lexical Vectorizer...")
        X_train_lexical = self.lexical_vectorizer.fit_transform(train_lexical)
        X_val_lexical = self.lexical_vectorizer.transform(val_lexical)
        X_test_lexical = self.lexical_vectorizer.transform(test_lexical)

        # 3. Community View (Subreddits only)
        train_community = self._process_subreddits_to_string(train_df)
        val_community = self._process_subreddits_to_string(val_df)
        test_community = self._process_subreddits_to_string(test_df)

        print("Fitting Community Vectorizer...")
        X_train_community = self.community_vectorizer.fit_transform(train_community)
        X_val_community = self.community_vectorizer.transform(val_community)
        X_test_community = self.community_vectorizer.transform(test_community)

        return (
            (X_train_holistic, X_val_holistic, X_test_holistic),
            (X_train_lexical, X_val_lexical, X_test_lexical),
            (X_train_community, X_val_community, X_test_community),
        )

    def _create_semantic_features(self, train_df, val_df, test_df):
        """Generates dense embedding features."""
        self._load_embedding_model()
        print("Generating Dense Semantic Embeddings...")

        def get_text_for_embedding(df):
            # Title + Body
            return (
                (
                    df["request_title"].fillna("")
                    + " "
                    + df["request_text_edit_aware"].fillna("")
                )
                .astype(str)
                .tolist()
            )

        train_texts = get_text_for_embedding(train_df)
        val_texts = get_text_for_embedding(val_df)
        test_texts = get_text_for_embedding(test_df)

        # Encode
        X_train_semantic = self.embedding_model.encode(
            train_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        X_val_semantic = self.embedding_model.encode(
            val_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        X_test_semantic = self.embedding_model.encode(
            test_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        return X_train_semantic, X_val_semantic, X_test_semantic

    def _get_metadata_features(self, df):
        """Extracts pre-processed metadata columns."""
        return df[config.METADATA_COLS].values.astype(np.float32)

    def generate_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main method to generate all feature views.
        Handles caching to avoid re-computation.
        """
        train_cache_path = os.path.join(self.cache_dir, "features_train.npz")
        val_cache_path = os.path.join(self.cache_dir, "features_val.npz")
        test_cache_path = os.path.join(self.cache_dir, "features_test.npz")

        # 1. Try Load from Cache
        if load_cached_data:
            if (
                os.path.exists(train_cache_path)
                and os.path.exists(val_cache_path)
                and os.path.exists(test_cache_path)
            ):
                print("Loading features from cache...")
                try:
                    train_data = dict(np.load(train_cache_path))
                    val_data = dict(np.load(val_cache_path))
                    test_data = dict(np.load(test_cache_path))
                    return train_data, val_data, test_data
                except Exception as e:
                    print(f"Error loading feature cache: {e}. Recomputing...")
            else:
                print("Feature cache not found. Computing from scratch...")

        # 2. Compute Features

        # A. Sparse Features
        (
            (X_train_hol, X_val_hol, X_test_hol),
            (X_train_lex, X_val_lex, X_test_lex),
            (X_train_com, X_val_com, X_test_com),
        ) = self._create_text_features(train_df, val_df, test_df)

        # B. Dense Semantic Features
        X_train_sem, X_val_sem, X_test_sem = self._create_semantic_features(
            train_df, val_df, test_df
        )

        # C. Metadata Features
        X_train_meta = self._get_metadata_features(train_df)
        X_val_meta = self._get_metadata_features(val_df)
        X_test_meta = self._get_metadata_features(test_df)

        # D. Targets
        y_train = train_df[config.TARGET_COL].values.astype(int)
        y_val = val_df[config.TARGET_COL].values.astype(int)
        # Test has no target, placeholder
        y_test = np.zeros(len(test_df))

        # 3. Concatenation (Adding Metadata to all views)
        # We convert sparse matrices to dense for saving/loading simplicity given high RAM.
        # This aligns with the requirement to use .npz and avoids pickling sparse objects.

        print("Concatenating Metadata and densifying sparse matrices...")

        def concat_and_densify(sparse_mat, dense_meta):
            # Convert sparse to dense
            dense_mat = sparse_mat.toarray().astype(np.float32)
            return np.hstack([dense_mat, dense_meta])

        def concat_dense(dense_emb, dense_meta):
            return np.hstack([dense_emb, dense_meta])

        # Train
        train_data = {
            "holistic": concat_and_densify(X_train_hol, X_train_meta),
            "lexical": concat_and_densify(X_train_lex, X_train_meta),
            "community": concat_and_densify(X_train_com, X_train_meta),
            "semantic": concat_dense(X_train_sem, X_train_meta),
            "metadata": X_train_meta,
            "y": y_train,
        }

        # Val
        val_data = {
            "holistic": concat_and_densify(X_val_hol, X_val_meta),
            "lexical": concat_and_densify(X_val_lex, X_val_meta),
            "community": concat_and_densify(X_val_com, X_val_meta),
            "semantic": concat_dense(X_val_sem, X_val_meta),
            "metadata": X_val_meta,
            "y": y_val,
        }

        # Test
        test_data = {
            "holistic": concat_and_densify(X_test_hol, X_test_meta),
            "lexical": concat_and_densify(X_test_lex, X_test_meta),
            "community": concat_and_densify(X_test_com, X_test_meta),
            "semantic": concat_dense(X_test_sem, X_test_meta),
            "metadata": X_test_meta,
            "y": y_test,  # Placeholder
        }

        # 4. Save to Cache
        print(f"Saving features to {self.cache_dir}...")
        utils.save_numpy_compressed(train_data, train_cache_path)
        utils.save_numpy_compressed(val_data, val_cache_path)
        utils.save_numpy_compressed(test_data, test_cache_path)

        return train_data, val_data, test_data
