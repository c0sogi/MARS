import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import get_processed_data


class FeatureEngine:
    def __init__(self):
        # Initialize transformers

        # 1. Metadata Transformers
        self.imputer = SimpleImputer(strategy="median")
        self.scaler_metadata = StandardScaler()

        # 2. Lexical Transformer (Sparse)
        self.tfidf_lexical = TfidfVectorizer(**Config.TFIDF_PARAMS)

        # 3. Behavioral Transformer (Sparse)
        self.tfidf_behavioral = TfidfVectorizer(**Config.TFIDF_PARAMS)

        # 4. Semantic Transformer (Dense)
        self.embedding_model = None  # Lazy initialization
        self.scaler_semantic = StandardScaler()

        # 5. Manifold Transformer (Dense PCA)
        self.pca = PCA(
            n_components=Config.PCA_COMPONENTS, random_state=Config.RANDOM_STATE
        )
        self.scaler_manifold = StandardScaler()

    def _get_embedding_model(self):
        """Lazy loader for the heavy embedding model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        return self.embedding_model

    def _get_cache_paths(self, split):
        """Generates file paths for all feature views for a given split."""
        base = Config.CACHE_DIR
        return {
            "metadata": os.path.join(base, f"{split}_metadata.npy"),
            "lexical": os.path.join(base, f"{split}_lexical.npz"),
            "behavioral": os.path.join(base, f"{split}_behavioral.npz"),
            "semantic_raw": os.path.join(
                base, f"{split}_semantic_raw.npy"
            ),  # Intermediate cache
            "semantic": os.path.join(base, f"{split}_semantic.npy"),
            "manifold": os.path.join(base, f"{split}_manifold.npy"),
            "target": os.path.join(base, f"{split}_target.npy"),
        }

    def _compute_raw_embeddings(self, texts, cache_path, load_cached_data):
        """
        Computes or loads raw SentenceTransformer embeddings.
        This is the most expensive operation, so caching is critical here.
        """
        if load_cached_data and os.path.exists(cache_path):
            return np.load(cache_path)

        model = self._get_embedding_model()
        # Encode with progress bar disabled for cleaner logs
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Save to cache
        np.save(cache_path, embeddings)
        return embeddings

    def fit_transform(
        self, split="train", load_cached_data=True, debug_sample_size=None
    ):
        """
        Fits transformers on the data and returns the transformed features.
        Always re-fits the lightweight sklearn objects (TF-IDF, Scalers, PCA) to ensure
        state consistency, but uses cached heavy embeddings if available.
        """
        # Load processed dataframe
        df = get_processed_data(split, load_cached_data, debug_sample_size)
        paths = self._get_cache_paths(split)

        # --- Target Variable ---
        y = None
        if Config.TARGET_COL in df.columns:
            y = df[Config.TARGET_COL].values
            np.save(paths["target"], y)

        # --- 1. Metadata View ---
        # Select allow-listed numerical features
        meta_df = df[Config.NUMERICAL_FEATURES].copy()
        # Impute missing values
        meta_imputed = self.imputer.fit_transform(meta_df)
        # Scale
        X_metadata = self.scaler_metadata.fit_transform(meta_imputed)
        np.save(paths["metadata"], X_metadata)

        # --- 2. Lexical View (Sparse) ---
        text_data = df[Config.TEXT_COL].fillna("").astype(str).tolist()
        X_lexical = self.tfidf_lexical.fit_transform(text_data)
        sp.save_npz(paths["lexical"], X_lexical)

        # --- 3. Behavioral View (Sparse) ---
        sub_data = df[Config.SUBREDDIT_COL].fillna("").astype(str).tolist()
        X_behavioral = self.tfidf_behavioral.fit_transform(sub_data)
        sp.save_npz(paths["behavioral"], X_behavioral)

        # --- 4. Semantic View (Dense) ---
        # Compute or Load Raw Embeddings (Heavy Operation)
        raw_embeddings = self._compute_raw_embeddings(
            text_data, paths["semantic_raw"], load_cached_data
        )

        # Scale for the Dense Semantic Branch
        X_semantic = self.scaler_semantic.fit_transform(raw_embeddings)
        np.save(paths["semantic"], X_semantic)

        # --- 5. Manifold View (Dense PCA) ---
        # Apply PCA to raw embeddings, then scale the components
        X_pca = self.pca.fit_transform(raw_embeddings)
        X_manifold = self.scaler_manifold.fit_transform(X_pca)
        np.save(paths["manifold"], X_manifold)

        return {
            "metadata": X_metadata,
            "lexical": X_lexical,
            "behavioral": X_behavioral,
            "semantic": X_semantic,
            "manifold": X_manifold,
        }, y

    def transform(self, split, load_cached_data=True, debug_sample_size=None):
        """
        Transforms data using the transformers fitted in fit_transform.
        Uses caching to avoid re-computation if the final feature files exist.
        """
        # Load processed dataframe
        df = get_processed_data(split, load_cached_data, debug_sample_size)
        paths = self._get_cache_paths(split)

        # --- Target Variable ---
        y = None
        if Config.TARGET_COL in df.columns:
            y = df[Config.TARGET_COL].values
            np.save(paths["target"], y)

        # Check if all final feature files exist in cache
        all_cached = (
            os.path.exists(paths["metadata"])
            and os.path.exists(paths["lexical"])
            and os.path.exists(paths["behavioral"])
            and os.path.exists(paths["semantic"])
            and os.path.exists(paths["manifold"])
        )

        if load_cached_data and all_cached:
            # Load features directly from cache
            X_metadata = np.load(paths["metadata"])
            X_lexical = sp.load_npz(paths["lexical"])
            X_behavioral = sp.load_npz(paths["behavioral"])
            X_semantic = np.load(paths["semantic"])
            X_manifold = np.load(paths["manifold"])
            return {
                "metadata": X_metadata,
                "lexical": X_lexical,
                "behavioral": X_behavioral,
                "semantic": X_semantic,
                "manifold": X_manifold,
            }, y

        # If not cached, compute using fitted transformers

        # --- 1. Metadata View ---
        meta_df = df[Config.NUMERICAL_FEATURES].copy()
        meta_imputed = self.imputer.transform(meta_df)
        X_metadata = self.scaler_metadata.transform(meta_imputed)
        np.save(paths["metadata"], X_metadata)

        # --- 2. Lexical View ---
        text_data = df[Config.TEXT_COL].fillna("").astype(str).tolist()
        X_lexical = self.tfidf_lexical.transform(text_data)
        sp.save_npz(paths["lexical"], X_lexical)

        # --- 3. Behavioral View ---
        sub_data = df[Config.SUBREDDIT_COL].fillna("").astype(str).tolist()
        X_behavioral = self.tfidf_behavioral.transform(sub_data)
        sp.save_npz(paths["behavioral"], X_behavioral)

        # --- 4. Semantic View ---
        # Compute or Load Raw Embeddings
        raw_embeddings = self._compute_raw_embeddings(
            text_data, paths["semantic_raw"], load_cached_data
        )
        X_semantic = self.scaler_semantic.transform(raw_embeddings)
        np.save(paths["semantic"], X_semantic)

        # --- 5. Manifold View ---
        X_pca = self.pca.transform(raw_embeddings)
        X_manifold = self.scaler_manifold.transform(X_pca)
        np.save(paths["manifold"], X_manifold)

        return {
            "metadata": X_metadata,
            "lexical": X_lexical,
            "behavioral": X_behavioral,
            "semantic": X_semantic,
            "manifold": X_manifold,
        }, y
