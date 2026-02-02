import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import Timer


class FeaturePipeline:
    """
    Orchestrates feature generation for the Hex-View Hybrid-Topology Stacking architecture.
    Generates Lexical, Behavioral, Semantic, Manifold, and Metadata feature views.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # --- 1. Sparse Lexical Branch Transformers ---
        self.lexical_vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)

        # --- 2. Sparse Behavioral Branch Transformers ---
        # We use the same TF-IDF params for community history
        self.community_vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)

        # --- 3. Dense Semantic Branch Transformers ---
        self.embedding_model_name = Config.EMBEDDING_MODEL
        self.embedding_model = None  # Lazy initialization

        # --- 4. Manifold Branch Transformers ---
        self.pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)

        # --- 5. Contextual/Metadata Branch Transformers ---
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        self.dense_features = Config.DENSE_FEATURES

    def _get_embedding_model(self):
        """Lazily loads the SentenceTransformer model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self.embedding_model

    def _process_text(self, df):
        """Extracts and cleans request text."""
        return df[Config.TEXT_COL].fillna("").astype(str).tolist()

    def _process_history(self, df):
        """Joins subreddit lists into space-separated strings."""

        def join_subs(subs):
            if isinstance(subs, list):
                return " ".join(subs)
            return ""

        if "requester_subreddits_at_request" in df.columns:
            return df["requester_subreddits_at_request"].apply(join_subs).tolist()
        else:
            # Fallback if column missing (should not happen based on schema)
            return [""] * len(df)

    def _process_metadata(self, df):
        """Extracts, converts, and prepares numerical metadata."""
        # Select allow-listed features
        data = df[self.dense_features].copy()
        # Ensure numeric types
        data = data.apply(pd.to_numeric, errors="coerce")
        return data.values

    def _save_cache(self, data, name, feature_type):
        """Saves feature arrays to disk using appropriate formats."""
        path_base = os.path.join(self.cache_dir, f"{name}_{feature_type}")
        if sp.issparse(data):
            sp.save_npz(path_base + ".npz", data)
        else:
            np.save(path_base + ".npy", data)

    def _load_cache(self, name, feature_type):
        """Attempts to load feature arrays from disk."""
        path_base = os.path.join(self.cache_dir, f"{name}_{feature_type}")
        path_npz = path_base + ".npz"
        path_npy = path_base + ".npy"

        if os.path.exists(path_npz):
            return sp.load_npz(path_npz)
        elif os.path.exists(path_npy):
            return np.load(path_npy)
        return None

    def fit_transform(self, df, split_name="train", load_cached_data=True):
        """
        Fits transformers on the provided DataFrame and generates features.

        Strategy:
        - Lightweight transformers (TF-IDF, PCA, Scaler) are ALWAYS fitted to ensure
          internal state is valid for subsequent 'transform' calls.
        - Heavyweight operations (Embeddings) use caching to save time.
        """
        features = {}

        # --- 1. Lexical (Sparse) ---
        with Timer("Lexical Feature Generation (Fit)"):
            text_data = self._process_text(df)
            features["lexical"] = self.lexical_vectorizer.fit_transform(text_data)
            self._save_cache(features["lexical"], split_name, "lexical")

        # --- 2. Behavioral (Sparse) ---
        with Timer("Behavioral Feature Generation (Fit)"):
            history_data = self._process_history(df)
            features["behavioral"] = self.community_vectorizer.fit_transform(
                history_data
            )
            self._save_cache(features["behavioral"], split_name, "behavioral")

        # --- 3. Semantic (Dense) ---
        # This is the bottleneck, so we prioritize caching here.
        with Timer("Semantic Feature Generation"):
            cached = self._load_cache(split_name, "semantic")
            if load_cached_data and cached is not None:
                print(f"Loaded cached semantic features for {split_name}")
                features["semantic"] = cached
            else:
                model = self._get_embedding_model()
                text_data = self._process_text(df)
                # Encode
                features["semantic"] = model.encode(
                    text_data,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                self._save_cache(features["semantic"], split_name, "semantic")

        # --- 4. Manifold (Dense) ---
        # Depends on Semantic. We must fit PCA on the semantic features.
        with Timer("Manifold Feature Generation (Fit)"):
            features["manifold"] = self.pca.fit_transform(features["semantic"])
            self._save_cache(features["manifold"], split_name, "manifold")

        # --- 5. Metadata (Dense) ---
        with Timer("Metadata Feature Generation (Fit)"):
            meta_raw = self._process_metadata(df)
            meta_imputed = self.imputer.fit_transform(meta_raw)
            features["metadata"] = self.scaler.fit_transform(meta_imputed)
            self._save_cache(features["metadata"], split_name, "metadata")

        return features

    def transform(self, df, split_name="test", load_cached_data=True):
        """
        Generates features for new data using fitted transformers.
        Uses caching aggressively if available.
        """
        features = {}

        # --- 1. Lexical ---
        with Timer(f"Lexical Feature Generation (Transform {split_name})"):
            cached = self._load_cache(split_name, "lexical")
            if load_cached_data and cached is not None:
                features["lexical"] = cached
            else:
                text_data = self._process_text(df)
                features["lexical"] = self.lexical_vectorizer.transform(text_data)
                self._save_cache(features["lexical"], split_name, "lexical")

        # --- 2. Behavioral ---
        with Timer(f"Behavioral Feature Generation (Transform {split_name})"):
            cached = self._load_cache(split_name, "behavioral")
            if load_cached_data and cached is not None:
                features["behavioral"] = cached
            else:
                history_data = self._process_history(df)
                features["behavioral"] = self.community_vectorizer.transform(
                    history_data
                )
                self._save_cache(features["behavioral"], split_name, "behavioral")

        # --- 3. Semantic ---
        with Timer(f"Semantic Feature Generation (Transform {split_name})"):
            cached = self._load_cache(split_name, "semantic")
            if load_cached_data and cached is not None:
                features["semantic"] = cached
            else:
                model = self._get_embedding_model()
                text_data = self._process_text(df)
                features["semantic"] = model.encode(
                    text_data,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                self._save_cache(features["semantic"], split_name, "semantic")

        # --- 4. Manifold ---
        with Timer(f"Manifold Feature Generation (Transform {split_name})"):
            cached = self._load_cache(split_name, "manifold")
            if load_cached_data and cached is not None:
                features["manifold"] = cached
            else:
                features["manifold"] = self.pca.transform(features["semantic"])
                self._save_cache(features["manifold"], split_name, "manifold")

        # --- 5. Metadata ---
        with Timer(f"Metadata Feature Generation (Transform {split_name})"):
            cached = self._load_cache(split_name, "metadata")
            if load_cached_data and cached is not None:
                features["metadata"] = cached
            else:
                meta_raw = self._process_metadata(df)
                meta_imputed = self.imputer.transform(meta_raw)
                features["metadata"] = self.scaler.transform(meta_imputed)
                self._save_cache(features["metadata"], split_name, "metadata")

        return features
