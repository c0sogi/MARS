import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import set_seed


class FeaturePipeline:
    """
    Hygienic Feature Engineering Pipeline for Deca-View Ensemble.
    Manages Lexical, Behavioral, Semantic, and Metadata feature generation with strict caching.
    """

    def __init__(self):
        set_seed()

        # 1. Sparse Lexical Branch (Text Modality)
        # TF-IDF on concatenated Title + Body
        self.lexical_vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)

        # 2. Sparse Behavioral Branch (History Modality)
        # TF-IDF on Subreddit History (Bag-of-Concepts)
        self.behavioral_vectorizer = TfidfVectorizer(
            max_features=Config.COMMUNITY_VOCAB_SIZE,
            binary=True,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",  # Simple tokenization
            stop_words="english",
        )

        # 3. Contextual Branch (Metadata Modality)
        # Imputation and Scaling for dense metadata
        self.metadata_imputer = SimpleImputer(strategy="median")
        self.metadata_scaler = StandardScaler()

        # 4. Dense Semantic Branch (Text Modality)
        # Lazy initialization for Sentence Transformer
        self.embedding_model = None

    def _get_paths(self, split_name):
        """Generates cache file paths for a given split."""
        base = os.path.join(Config.CACHE_DIR, f"{split_name}")
        return {
            "lexical": f"{base}_lexical.npz",
            "behavioral": f"{base}_behavioral.npz",
            "semantic": f"{base}_semantic.npy",
            "metadata": f"{base}_metadata.npy",
        }

    def _process_text(self, df):
        """Concatenates title and edit-aware body text."""
        title = df[Config.TITLE_COL].fillna("").astype(str)
        body = df[Config.TEXT_COL].fillna("").astype(str)
        return (title + " " + body).tolist()

    def _process_community(self, df):
        """Joins subreddit lists into space-separated strings."""

        def join_subs(x):
            if isinstance(x, list):
                return " ".join(x)
            elif isinstance(x, np.ndarray):
                return " ".join(x)
            return str(x) if x is not None else ""

        return df[Config.COMMUNITY_COL].apply(join_subs).tolist()

    def _build_semantic(self, texts, split_name, load_cached_data):
        """Generates or loads dense semantic embeddings."""
        path = self._get_paths(split_name)["semantic"]

        if load_cached_data and os.path.exists(path):
            return np.load(path)

        # Compute if not cached
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            # Move to GPU if available
            import torch

            if torch.cuda.is_available():
                self.embedding_model.to("cuda")

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Save to cache
        np.save(path, embeddings)
        return embeddings

    def fit_transform(self, df, split_name="train", load_cached_data=True):
        """
        Fits vectorizers/scalers and transforms the data.
        Note: We ALWAYS fit the lightweight components (TF-IDF, Scaler) on the input df
        to ensure the pipeline state is correct for future inference, even if we load
        the transformed matrices from cache.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        paths = self._get_paths(split_name)

        # --- 1. Lexical ---
        raw_text = self._process_text(df)
        self.lexical_vectorizer.fit(raw_text)

        if load_cached_data and os.path.exists(paths["lexical"]):
            X_lex = sp.load_npz(paths["lexical"])
        else:
            X_lex = self.lexical_vectorizer.transform(raw_text)
            sp.save_npz(paths["lexical"], X_lex)

        # --- 2. Behavioral ---
        raw_community = self._process_community(df)
        self.behavioral_vectorizer.fit(raw_community)

        if load_cached_data and os.path.exists(paths["behavioral"]):
            X_beh = sp.load_npz(paths["behavioral"])
        else:
            X_beh = self.behavioral_vectorizer.transform(raw_community)
            sp.save_npz(paths["behavioral"], X_beh)

        # --- 3. Metadata ---
        meta_df = df[Config.METADATA_COLS].copy()
        self.metadata_imputer.fit(meta_df)
        meta_imputed = self.metadata_imputer.transform(meta_df)
        self.metadata_scaler.fit(meta_imputed)

        if load_cached_data and os.path.exists(paths["metadata"]):
            X_meta = np.load(paths["metadata"])
        else:
            X_meta = self.metadata_scaler.transform(meta_imputed)
            np.save(paths["metadata"], X_meta)

        # --- 4. Semantic ---
        # No 'fit' required for pre-trained model, but we need raw_text
        X_sem = self._build_semantic(raw_text, split_name, load_cached_data)

        return {
            "lexical": X_lex,
            "behavioral": X_beh,
            "metadata": X_meta,
            "semantic": X_sem,
        }

    def transform(self, df, split_name, load_cached_data=True):
        """
        Transforms data using pre-fitted components.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        paths = self._get_paths(split_name)

        # --- 1. Lexical ---
        if load_cached_data and os.path.exists(paths["lexical"]):
            X_lex = sp.load_npz(paths["lexical"])
        else:
            raw_text = self._process_text(df)
            X_lex = self.lexical_vectorizer.transform(raw_text)
            sp.save_npz(paths["lexical"], X_lex)

        # --- 2. Behavioral ---
        if load_cached_data and os.path.exists(paths["behavioral"]):
            X_beh = sp.load_npz(paths["behavioral"])
        else:
            raw_community = self._process_community(df)
            X_beh = self.behavioral_vectorizer.transform(raw_community)
            sp.save_npz(paths["behavioral"], X_beh)

        # --- 3. Metadata ---
        if load_cached_data and os.path.exists(paths["metadata"]):
            X_meta = np.load(paths["metadata"])
        else:
            meta_df = df[Config.METADATA_COLS].copy()
            meta_imputed = self.metadata_imputer.transform(meta_df)
            X_meta = self.metadata_scaler.transform(meta_imputed)
            np.save(paths["metadata"], X_meta)

        # --- 4. Semantic ---
        if load_cached_data and os.path.exists(paths["semantic"]):
            X_sem = np.load(paths["semantic"])
        else:
            raw_text = self._process_text(df)
            # Force compute if not cached
            X_sem = self._build_semantic(raw_text, split_name, load_cached_data=False)

        return {
            "lexical": X_lex,
            "behavioral": X_beh,
            "metadata": X_meta,
            "semantic": X_sem,
        }
