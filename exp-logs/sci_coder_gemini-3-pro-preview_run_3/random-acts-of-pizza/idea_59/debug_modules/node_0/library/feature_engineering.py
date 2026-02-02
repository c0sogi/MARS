import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import (
    WORKING_DIR,
    SEED,
    ALLOW_LIST,
    LEXICAL_VECTORIZER_PARAMS,
    COMMUNITY_VECTORIZER_PARAMS,
    TEXT_COLS,
    COMMUNITY_COL,
    EMBEDDING_MODEL,
)
from library.utils import get_logger, save_cache, load_cache

# Initialize Logger
logger = get_logger("feature_engineering")


class FeaturePipeline:
    def __init__(self):
        # Initialize transformers with fixed seeds where applicable
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.lexical_vectorizer = TfidfVectorizer(**LEXICAL_VECTORIZER_PARAMS)
        self.community_vectorizer = TfidfVectorizer(**COMMUNITY_VECTORIZER_PARAMS)
        self.embedding_model = None  # Lazy load to save resources if loading from cache

    def _get_embedding_model(self):
        """Lazy loader for the heavy embedding model."""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self.embedding_model

    def _save_transformers(self):
        """Persist fitted transformers to disk."""
        path = os.path.join(WORKING_DIR, "transformers.joblib")
        joblib.dump(
            {
                "imputer": self.imputer,
                "scaler": self.scaler,
                "lexical_vectorizer": self.lexical_vectorizer,
                "community_vectorizer": self.community_vectorizer,
            },
            path,
        )
        logger.info("Transformers saved to disk.")

    def _load_transformers(self):
        """Load fitted transformers from disk."""
        path = os.path.join(WORKING_DIR, "transformers.joblib")
        if os.path.exists(path):
            data = joblib.load(path)
            self.imputer = data["imputer"]
            self.scaler = data["scaler"]
            self.lexical_vectorizer = data["lexical_vectorizer"]
            self.community_vectorizer = data["community_vectorizer"]
            return True
        return False

    def _wrap_sparse(self, matrix):
        """Wrap sparse matrix in object array for compatibility with utils.save_cache."""
        arr = np.empty(1, dtype=object)
        arr[0] = matrix
        return arr

    def _unwrap_sparse(self, arr):
        """Unwrap sparse matrix from object array."""
        return arr.item()

    def build_metadata_features(self, df, fit=False):
        """Extracts, imputes, and scales allow-listed metadata features."""
        # Select allow-listed columns that exist in the dataframe
        selected_cols = [c for c in ALLOW_LIST if c in df.columns]

        # Create a copy to avoid SettingWithCopy warnings
        X = df[selected_cols].copy()

        # Impute
        if fit:
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = self.imputer.transform(X)

        # Scale
        if fit:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)

        return X_scaled.astype(np.float32)

    def build_lexical_features(self, df, fit=False):
        """Vectorizes concatenated title and body text."""
        # Concatenate text columns safely
        text_data = df[TEXT_COLS[0]].fillna("") + " " + df[TEXT_COLS[1]].fillna("")

        if fit:
            X = self.lexical_vectorizer.fit_transform(text_data)
        else:
            X = self.lexical_vectorizer.transform(text_data)

        return X

    def build_behavioral_features(self, df, fit=False):
        """Vectorizes subreddit history as a bag-of-concepts."""

        def join_subreddits(x):
            if isinstance(x, list):
                return " ".join(x)
            elif isinstance(x, np.ndarray):
                return " ".join(x)
            return str(x) if pd.notnull(x) else ""

        corpus = df[COMMUNITY_COL].apply(join_subreddits)

        if fit:
            X = self.community_vectorizer.fit_transform(corpus)
        else:
            X = self.community_vectorizer.transform(corpus)

        return X

    def build_dense_features(self, df):
        """Generates semantic embeddings."""
        text_data = (
            df[TEXT_COLS[0]].fillna("") + " " + df[TEXT_COLS[1]].fillna("")
        ).tolist()

        model = self._get_embedding_model()
        # encode returns numpy array by default if convert_to_numpy=True (default)
        embeddings = model.encode(text_data, show_progress_bar=False)

        return embeddings.astype(np.float32)

    def fit_transform(self, df, load_cached_data=True):
        """
        Fits transformers on the provided dataframe (Union Dataset) and returns feature matrices.
        Uses caching to avoid re-computation.
        """
        cache_keys = {
            "metadata": "X_train_meta",
            "lexical": "X_train_lexical",
            "behavioral": "X_train_behavioral",
            "semantic": "X_train_semantic",
        }

        # 1. Try loading from cache
        if load_cached_data:
            cached_data = {}
            all_exist = True

            # Check if transformers exist
            if not self._load_transformers():
                all_exist = False

            # Check if data files exist
            if all_exist:
                for key, filename in cache_keys.items():
                    data = load_cache(filename, WORKING_DIR)
                    if data is None:
                        all_exist = False
                        break

                    # Unwrap sparse matrices
                    if key in ["lexical", "behavioral"]:
                        cached_data[key] = self._unwrap_sparse(data)
                    else:
                        cached_data[key] = data

            if all_exist:
                logger.info("Loaded all training features and transformers from cache.")
                return cached_data

        # 2. Compute from scratch
        logger.info("Computing training features from scratch...")

        features = {}
        features["metadata"] = self.build_metadata_features(df, fit=True)
        features["lexical"] = self.build_lexical_features(df, fit=True)
        features["behavioral"] = self.build_behavioral_features(df, fit=True)
        features["semantic"] = self.build_dense_features(df)

        # 3. Save Transformers
        self._save_transformers()

        # 4. Save Data to Cache
        save_cache(features["metadata"], cache_keys["metadata"], WORKING_DIR)
        save_cache(
            self._wrap_sparse(features["lexical"]), cache_keys["lexical"], WORKING_DIR
        )
        save_cache(
            self._wrap_sparse(features["behavioral"]),
            cache_keys["behavioral"],
            WORKING_DIR,
        )
        save_cache(features["semantic"], cache_keys["semantic"], WORKING_DIR)

        return features

    def transform(self, df, load_cached_data=True):
        """
        Transforms the provided dataframe (Test Dataset) using fitted transformers.
        Uses caching to avoid re-computation.
        """
        cache_keys = {
            "metadata": "X_test_meta",
            "lexical": "X_test_lexical",
            "behavioral": "X_test_behavioral",
            "semantic": "X_test_semantic",
        }

        # 1. Try loading from cache
        if load_cached_data:
            cached_data = {}
            all_exist = True

            for key, filename in cache_keys.items():
                data = load_cache(filename, WORKING_DIR)
                if data is None:
                    all_exist = False
                    break

                # Unwrap sparse matrices
                if key in ["lexical", "behavioral"]:
                    cached_data[key] = self._unwrap_sparse(data)
                else:
                    cached_data[key] = data

            if all_exist:
                # Ensure transformers are loaded for consistency
                self._load_transformers()
                logger.info("Loaded all test features from cache.")
                return cached_data

        # 2. Compute from scratch
        logger.info("Computing test features from scratch...")

        # Ensure transformers are loaded
        if not self._load_transformers():
            # Check if in-memory transformers are fitted (e.g. if run in same script)
            try:
                # Quick check on scaler
                _ = self.scaler.mean_
            except AttributeError:
                raise RuntimeError(
                    "Transformers not fitted! Run fit_transform on training data first."
                )

        features = {}
        features["metadata"] = self.build_metadata_features(df, fit=False)
        features["lexical"] = self.build_lexical_features(df, fit=False)
        features["behavioral"] = self.build_behavioral_features(df, fit=False)
        features["semantic"] = self.build_dense_features(df)

        # 3. Save Data to Cache
        save_cache(features["metadata"], cache_keys["metadata"], WORKING_DIR)
        save_cache(
            self._wrap_sparse(features["lexical"]), cache_keys["lexical"], WORKING_DIR
        )
        save_cache(
            self._wrap_sparse(features["behavioral"]),
            cache_keys["behavioral"],
            WORKING_DIR,
        )
        save_cache(features["semantic"], cache_keys["semantic"], WORKING_DIR)

        return features
