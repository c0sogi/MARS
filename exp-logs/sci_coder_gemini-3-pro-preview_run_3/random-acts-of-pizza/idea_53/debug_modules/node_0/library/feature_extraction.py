import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import (
    CACHE_DIR,
    METADATA_FEATURES,
    LEXICAL_VECTORIZER_PARAMS,
    COMMUNITY_VECTORIZER_PARAMS,
    EMBEDDING_MODEL_NAME,
    RANDOM_SEED,
)


class FeaturePipeline:
    """
    Manages feature extraction, transformation, and caching for the multi-modal pipeline.
    Generates four feature matrices:
    1. X_lexical: Sparse TF-IDF of concatenated text.
    2. X_community: Sparse Bag-of-Concepts of subreddit history.
    3. X_semantic: Dense embeddings of concatenated text.
    4. X_meta: Scaled and imputed numerical metadata.
    """

    def __init__(self, load_cached_data=True):
        self.load_cached_data = load_cached_data

        # Initialize Transformers with config parameters
        self.lexical_vectorizer = TfidfVectorizer(**LEXICAL_VECTORIZER_PARAMS)
        self.community_vectorizer = CountVectorizer(**COMMUNITY_VECTORIZER_PARAMS)

        # Metadata preprocessing: Median Imputation + Standard Scaling
        self.meta_imputer = SimpleImputer(strategy="median")
        self.meta_scaler = StandardScaler()

        # Embedding model is loaded lazily to optimize runtime if cache exists
        self.embedding_model = None

    def _get_cache_paths(self, split):
        """Returns a dictionary of cache file paths for a given split."""
        return {
            "lexical": os.path.join(CACHE_DIR, f"X_{split}_lexical.npz"),
            "community": os.path.join(CACHE_DIR, f"X_{split}_community.npz"),
            "semantic": os.path.join(CACHE_DIR, f"X_{split}_semantic.npy"),
            "meta": os.path.join(CACHE_DIR, f"X_{split}_meta.npy"),
        }

    def _load_from_cache(self, split):
        """Attempts to load all feature matrices for a split from cache."""
        paths = self._get_cache_paths(split)

        if not self.load_cached_data:
            return None

        # Check if all required files exist
        if all(os.path.exists(p) for p in paths.values()):
            try:
                X_lexical = scipy.sparse.load_npz(paths["lexical"])
                X_community = scipy.sparse.load_npz(paths["community"])
                X_semantic = np.load(paths["semantic"])
                X_meta = np.load(paths["meta"])
                return X_lexical, X_community, X_semantic, X_meta
            except Exception as e:
                print(f"Warning: Failed to load cache for {split} ({e}). Recomputing.")
                return None
        return None

    def _save_to_cache(self, split, X_lexical, X_community, X_semantic, X_meta):
        """Saves all feature matrices to cache."""
        paths = self._get_cache_paths(split)
        os.makedirs(CACHE_DIR, exist_ok=True)

        scipy.sparse.save_npz(paths["lexical"], X_lexical)
        scipy.sparse.save_npz(paths["community"], X_community)
        np.save(paths["semantic"], X_semantic)
        np.save(paths["meta"], X_meta)

    def _get_embedding_model(self):
        """Lazily loads the SentenceTransformer model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self.embedding_model

    def fit_transform(self, df):
        """
        Fits vectorizers and scalers on the training DataFrame and returns transformed features.

        NOTE: This method ALWAYS fits the lightweight sklearn models (TF-IDF, Scaler)
        on the provided dataframe, even if the output matrices are loaded from cache.
        This ensures the pipeline's internal state is correctly set up to transform
        subsequent validation/test sets that might not be cached.
        """
        split = "train"

        # 1. Fit lightweight models
        # Lexical
        self.lexical_vectorizer.fit(df["text_combined"])

        # Community
        self.community_vectorizer.fit(df["subreddit_string"])

        # Metadata
        meta_data = df[METADATA_FEATURES].values
        self.meta_imputer.fit(meta_data)
        meta_imputed = self.meta_imputer.transform(meta_data)
        self.meta_scaler.fit(meta_imputed)

        # 2. Check Cache for computed matrices
        cached = self._load_from_cache(split)
        if cached is not None:
            return cached

        # 3. Compute features if not cached
        # Lexical
        X_lexical = self.lexical_vectorizer.transform(df["text_combined"])

        # Community
        X_community = self.community_vectorizer.transform(df["subreddit_string"])

        # Metadata
        X_meta = self.meta_scaler.transform(meta_imputed)

        # Semantic (Heavy computation)
        model = self._get_embedding_model()
        X_semantic = model.encode(
            df["text_combined"].tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 4. Save to cache
        self._save_to_cache(split, X_lexical, X_community, X_semantic, X_meta)

        return X_lexical, X_community, X_semantic, X_meta

    def transform(self, df, split):
        """
        Transforms a DataFrame (val/test) using the previously fitted models.
        Checks cache first to avoid redundant computation.
        """
        # 1. Check Cache
        cached = self._load_from_cache(split)
        if cached is not None:
            return cached

        # 2. Compute features
        # Lexical
        X_lexical = self.lexical_vectorizer.transform(df["text_combined"])

        # Community
        X_community = self.community_vectorizer.transform(df["subreddit_string"])

        # Metadata
        meta_data = df[METADATA_FEATURES].values
        meta_imputed = self.meta_imputer.transform(meta_data)
        X_meta = self.meta_scaler.transform(meta_imputed)

        # Semantic
        model = self._get_embedding_model()
        X_semantic = model.encode(
            df["text_combined"].tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 3. Save to cache
        self._save_to_cache(split, X_lexical, X_community, X_semantic, X_meta)

        return X_lexical, X_community, X_semantic, X_meta
