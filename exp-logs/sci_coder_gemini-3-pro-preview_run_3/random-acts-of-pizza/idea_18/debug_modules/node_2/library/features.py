import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config


class MultiModalFeatureGenerator:
    """
    Generates features for the Symmetric Multi-Modal Stacking Ensemble.
    Handles Text (Sparse/Dense), Behavioral (Sparse/Dense), and Contextual (Dense) modalities.
    Implements per-split caching to optimize runtime.
    """

    def __init__(self):
        # Contextual / Metadata Processors
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # Text Modality Processors
        self.text_tfidf = TfidfVectorizer(**Config.TEXT_TFIDF_PARAMS)

        # Behavioral Modality Processors
        self.behav_tfidf = TfidfVectorizer(**Config.SUBREDDIT_TFIDF_PARAMS)

        # Dense Embedding Model (Lazy loaded)
        self.embedding_model = None

    def _load_embedding_model(self):
        """Lazy loads the SentenceTransformer model to GPU."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                Config.MPNET_MODEL_NAME, device=Config.DEVICE
            )

    def _extract_dense_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts and engineers dense numerical features.
        Includes temporal feature engineering (Hour, Day) as per solution design.
        """
        # Select base numerical columns defined in Config
        data = df[Config.NUMERICAL_COLS].copy()

        # Temporal Feature Engineering
        if "unix_timestamp_of_request" in data.columns:
            dt = pd.to_datetime(data["unix_timestamp_of_request"], unit="s")
            data["request_hour"] = dt.dt.hour
            data["request_dayofweek"] = dt.dt.dayofweek

        return data.values

    def fit(self, df: pd.DataFrame):
        """
        Fits all stateful preprocessors (Imputer, Scaler, Vectorizers) on the training data.
        """
        # 1. Fit Metadata Processors
        meta_values = self._extract_dense_features(df)
        self.imputer.fit(meta_values)
        meta_imputed = self.imputer.transform(meta_values)
        self.scaler.fit(meta_imputed)

        # 2. Fit Text TF-IDF
        text_data = df[Config.TEXT_COL].fillna("").astype(str)
        self.text_tfidf.fit(text_data)

        # 3. Fit Behavioral TF-IDF
        # Assumes data_utils.preprocess_subreddits has converted list to string
        behav_data = df[Config.SUBREDDIT_COL].fillna("").astype(str)
        self.behav_tfidf.fit(behav_data)

        return self

    def _generate_embeddings(self, texts: list) -> np.ndarray:
        """Generates dense embeddings using the MPNet model."""
        self._load_embedding_model()
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=Config.BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def transform(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> dict:
        """
        Transforms the input DataFrame into the five feature sets required by the ensemble.
        Implements caching to avoid re-computation.

        Returns dictionary with keys: 'lexical', 'semantic', 'community', 'persona', 'meta'.
        """
        # Define cache file paths
        cache_files = {
            "lexical": os.path.join(Config.CACHE_DIR, f"{split_name}_lexical.npz"),
            "semantic": os.path.join(Config.CACHE_DIR, f"{split_name}_semantic.npy"),
            "community": os.path.join(Config.CACHE_DIR, f"{split_name}_community.npz"),
            "persona": os.path.join(Config.CACHE_DIR, f"{split_name}_persona.npy"),
            "meta": os.path.join(Config.CACHE_DIR, f"{split_name}_meta.npy"),
        }

        # Check if all cache files exist
        cache_hit = load_cached_data and all(
            os.path.exists(p) for p in cache_files.values()
        )

        if cache_hit:
            print(f"Loading {split_name} features from cache...")
            return {
                "lexical": sp.load_npz(cache_files["lexical"]),
                "semantic": np.load(cache_files["semantic"]),
                "community": sp.load_npz(cache_files["community"]),
                "persona": np.load(cache_files["persona"]),
                "meta": np.load(cache_files["meta"]),
            }

        print(f"Generating {split_name} features (Cache miss or force reload)...")
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # ---------------------------------------------------------
        # 1. Contextual Modality (Metadata)
        # ---------------------------------------------------------
        meta_raw = self._extract_dense_features(df)
        meta_imputed = self.imputer.transform(meta_raw)
        X_meta = self.scaler.transform(meta_imputed).astype(np.float32)

        # Create sparse version for concatenation with sparse matrices
        X_meta_sparse = sp.csr_matrix(X_meta)

        # ---------------------------------------------------------
        # 2. Text Modality
        # ---------------------------------------------------------
        text_series = df[Config.TEXT_COL].fillna("").astype(str)
        text_list = text_series.tolist()

        # A. Lexical (Sparse): TF-IDF + Metadata
        X_text_tfidf = self.text_tfidf.transform(text_series)
        X_lexical = sp.hstack([X_text_tfidf, X_meta_sparse], format="csr")

        # B. Semantic (Dense): MPNet Embeddings + Metadata
        X_text_emb = self._generate_embeddings(text_list)
        X_semantic = np.hstack([X_text_emb, X_meta])

        # ---------------------------------------------------------
        # 3. Behavioral Modality
        # ---------------------------------------------------------
        behav_series = df[Config.SUBREDDIT_COL].fillna("").astype(str)
        behav_list = behav_series.tolist()

        # A. Community (Sparse): Bag-of-Subreddits + Metadata
        X_behav_tfidf = self.behav_tfidf.transform(behav_series)
        X_community = sp.hstack([X_behav_tfidf, X_meta_sparse], format="csr")

        # B. Persona (Dense): Subreddit Embeddings + Metadata
        X_behav_emb = self._generate_embeddings(behav_list)
        X_persona = np.hstack([X_behav_emb, X_meta])

        # ---------------------------------------------------------
        # Save to Cache
        # ---------------------------------------------------------
        sp.save_npz(cache_files["lexical"], X_lexical)
        np.save(cache_files["semantic"], X_semantic)
        sp.save_npz(cache_files["community"], X_community)
        np.save(cache_files["persona"], X_persona)
        np.save(cache_files["meta"], X_meta)

        return {
            "lexical": X_lexical,
            "semantic": X_semantic,
            "community": X_community,
            "persona": X_persona,
            "meta": X_meta,
        }
