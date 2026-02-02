import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import (
    WORKING_DIR,
    CACHE_DIR,
    MODELS_DIR,
    METADATA_ALLOWLIST,
    TEXT_COLS,
    SUBREDDIT_COL,
    EMBEDDING_MODEL,
    LEXICAL_VECTORIZER_PARAMS,
    COMMUNITY_VECTORIZER_PARAMS,
    SEED,
)
from library.utils import Timer, save_joblib, load_joblib


class FeaturePipeline:
    """
    Manages feature engineering for the Hept-View Stacking Ensemble.
    Handles four modalities:
    1. Lexical (Sparse TF-IDF on Title + Body)
    2. Behavioral (Sparse TF-IDF on Subreddit History)
    3. Semantic (Dense Embeddings)
    4. Contextual (Dense Metadata)
    """

    def __init__(self):
        self.lexical_vectorizer = TfidfVectorizer(**LEXICAL_VECTORIZER_PARAMS)
        self.community_vectorizer = TfidfVectorizer(**COMMUNITY_VECTORIZER_PARAMS)
        self.meta_imputer = SimpleImputer(strategy="median")
        self.meta_scaler = StandardScaler()
        self.embedding_model = None  # Lazy loading

    def _get_paths(self, cache_name):
        """Generates file paths for cached feature matrices."""
        return {
            "lexical": os.path.join(CACHE_DIR, f"{cache_name}_lexical.npz"),
            "behavioral": os.path.join(CACHE_DIR, f"{cache_name}_behavioral.npz"),
            "semantic": os.path.join(CACHE_DIR, f"{cache_name}_semantic.npy"),
            "meta": os.path.join(CACHE_DIR, f"{cache_name}_meta.npy"),
        }

    def _get_model_paths(self):
        """Generates file paths for fitted transformers."""
        return {
            "lexical_vectorizer": os.path.join(MODELS_DIR, "lexical_vectorizer.joblib"),
            "community_vectorizer": os.path.join(
                MODELS_DIR, "community_vectorizer.joblib"
            ),
            "meta_imputer": os.path.join(MODELS_DIR, "meta_imputer.joblib"),
            "meta_scaler": os.path.join(MODELS_DIR, "meta_scaler.joblib"),
        }

    def _load_embedding_model(self):
        if self.embedding_model is None:
            with Timer("Loading SentenceTransformer"):
                # Use CPU/GPU automatically
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    def _process_text(self, df):
        """Concatenates title and edit-aware body text."""
        # Fill NA with empty string to avoid errors
        title = df[TEXT_COLS[0]].fillna("").astype(str)
        body = df[TEXT_COLS[1]].fillna("").astype(str)
        return title + " " + body

    def _process_subreddits(self, df):
        """Joins list of subreddits into a space-separated string."""
        # Handle cases where the column might be NaN or empty lists
        return df[SUBREDDIT_COL].apply(
            lambda x: " ".join(x) if isinstance(x, list) else ""
        )

    def _process_metadata(self, df, fit=False):
        """Extracts, imputes, and scales allow-listed metadata."""
        # Select allowed columns
        meta_df = df[METADATA_ALLOWLIST].copy()

        # Convert to float for safety
        meta_df = meta_df.astype(float)

        if fit:
            # Fit imputer and scaler
            meta_imputed = self.meta_imputer.fit_transform(meta_df)
            meta_scaled = self.meta_scaler.fit_transform(meta_imputed)
        else:
            # Transform
            meta_imputed = self.meta_imputer.transform(meta_df)
            meta_scaled = self.meta_scaler.transform(meta_imputed)

        return meta_scaled

    def fit_transform(self, df, load_cached_data=True, cache_name="train"):
        """
        Fits vectorizers/scalers and transforms the dataset.
        """
        paths = self._get_paths(cache_name)
        model_paths = self._get_model_paths()

        # Check cache
        if load_cached_data and all(os.path.exists(p) for p in paths.values()):
            with Timer(f"Loading cached features ({cache_name})"):
                return {
                    "lexical": sparse.load_npz(paths["lexical"]),
                    "behavioral": sparse.load_npz(paths["behavioral"]),
                    "semantic": np.load(paths["semantic"]),
                    "meta": np.load(paths["meta"]),
                }

        with Timer("Feature Engineering (Fit & Transform)"):
            # 1. Text Processing
            text_series = self._process_text(df)

            # 2. Lexical (Sparse)
            print("Vectorizing Lexical features...")
            X_lexical = self.lexical_vectorizer.fit_transform(text_series)

            # 3. Behavioral (Sparse)
            print("Vectorizing Behavioral features...")
            subreddit_series = self._process_subreddits(df)
            X_behavioral = self.community_vectorizer.fit_transform(subreddit_series)

            # 4. Semantic (Dense)
            print("Generating Semantic embeddings...")
            self._load_embedding_model()
            # Encode returns numpy array by default
            X_semantic = self.embedding_model.encode(
                text_series.tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # 5. Metadata (Dense)
            print("Processing Metadata...")
            X_meta = self._process_metadata(df, fit=True)

            # Save Fitted Models
            print("Saving fitted models...")
            save_joblib(self.lexical_vectorizer, model_paths["lexical_vectorizer"])
            save_joblib(self.community_vectorizer, model_paths["community_vectorizer"])
            save_joblib(self.meta_imputer, model_paths["meta_imputer"])
            save_joblib(self.meta_scaler, model_paths["meta_scaler"])

            # Save Feature Matrices
            print(f"Caching features to {CACHE_DIR}...")
            sparse.save_npz(paths["lexical"], X_lexical)
            sparse.save_npz(paths["behavioral"], X_behavioral)
            np.save(paths["semantic"], X_semantic)
            np.save(paths["meta"], X_meta)

        return {
            "lexical": X_lexical,
            "behavioral": X_behavioral,
            "semantic": X_semantic,
            "meta": X_meta,
        }

    def transform(self, df, load_cached_data=True, cache_name="test"):
        """
        Transforms a dataset using previously fitted models.
        """
        paths = self._get_paths(cache_name)
        model_paths = self._get_model_paths()

        # Check cache
        if load_cached_data and all(os.path.exists(p) for p in paths.values()):
            with Timer(f"Loading cached features ({cache_name})"):
                return {
                    "lexical": sparse.load_npz(paths["lexical"]),
                    "behavioral": sparse.load_npz(paths["behavioral"]),
                    "semantic": np.load(paths["semantic"]),
                    "meta": np.load(paths["meta"]),
                }

        with Timer("Feature Engineering (Transform Only)"):
            # Load models if not in memory (e.g. if this is a fresh run)
            # We check one attribute to see if it's fitted.
            # sklearn vectorizers have 'vocabulary_' attribute when fitted.
            if not hasattr(self.lexical_vectorizer, "vocabulary_"):
                print("Loading fitted models from disk...")
                self.lexical_vectorizer = load_joblib(model_paths["lexical_vectorizer"])
                self.community_vectorizer = load_joblib(
                    model_paths["community_vectorizer"]
                )
                self.meta_imputer = load_joblib(model_paths["meta_imputer"])
                self.meta_scaler = load_joblib(model_paths["meta_scaler"])

            # 1. Text Processing
            text_series = self._process_text(df)

            # 2. Lexical (Sparse)
            print("Vectorizing Lexical features...")
            X_lexical = self.lexical_vectorizer.transform(text_series)

            # 3. Behavioral (Sparse)
            print("Vectorizing Behavioral features...")
            subreddit_series = self._process_subreddits(df)
            X_behavioral = self.community_vectorizer.transform(subreddit_series)

            # 4. Semantic (Dense)
            print("Generating Semantic embeddings...")
            self._load_embedding_model()
            X_semantic = self.embedding_model.encode(
                text_series.tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # 5. Metadata (Dense)
            print("Processing Metadata...")
            X_meta = self._process_metadata(df, fit=False)

            # Save Feature Matrices
            print(f"Caching features to {CACHE_DIR}...")
            sparse.save_npz(paths["lexical"], X_lexical)
            sparse.save_npz(paths["behavioral"], X_behavioral)
            np.save(paths["semantic"], X_semantic)
            np.save(paths["meta"], X_meta)

        return {
            "lexical": X_lexical,
            "behavioral": X_behavioral,
            "semantic": X_semantic,
            "meta": X_meta,
        }
