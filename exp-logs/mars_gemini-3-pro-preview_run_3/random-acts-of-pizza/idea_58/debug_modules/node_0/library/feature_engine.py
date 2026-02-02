import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import set_seed


class GranularLexicalVectorizer:
    """
    Handles the extraction of sparse lexical features using a granular token pattern
    to preserve single-character signals (e.g., 'I', '$').
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(**Config.LEXICAL_VECTORIZER_PARAMS)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fits the vectorizer on the combined text column.
        """
        set_seed()
        text_data = df["text_combined"].fillna("").astype(str)
        self.vectorizer.fit(text_data)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """
        Transforms the text data into a sparse TF-IDF matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transform.")

        text_data = df["text_combined"].fillna("").astype(str)
        return self.vectorizer.transform(text_data)


class CommunityVectorizer:
    """
    Handles the extraction of sparse behavioral features from subreddit history.
    Treats subreddit participation as a 'Bag-of-Concepts'.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(**Config.COMMUNITY_VECTORIZER_PARAMS)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fits the vectorizer on the subreddit string column.
        """
        set_seed()
        sub_data = df["subreddit_string"].fillna("").astype(str)
        self.vectorizer.fit(sub_data)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        """
        Transforms the subreddit strings into a sparse matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted before transform.")

        sub_data = df["subreddit_string"].fillna("").astype(str)
        return self.vectorizer.transform(sub_data)


class MetadataScaler:
    """
    Handles the selection and scaling of numerical metadata features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.features = Config.META_FEATURES_ALLOWLIST
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fits the scaler on the allow-listed metadata columns.
        """
        set_seed()
        # Ensure all features exist, fill missing with 0 if somehow not handled upstream
        X = df[self.features].fillna(0).values
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms the metadata columns using the fitted scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform.")

        X = df[self.features].fillna(0).values
        return self.scaler.transform(X)


class SemanticEmbedder:
    """
    Handles the generation of dense semantic embeddings using a pre-trained Transformer.
    Implements strict caching to avoid redundant computation.
    """

    def __init__(self):
        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.cache_dir = Config.WORKING_DIR
        # Model is loaded lazily or upon first use to save resources if not needed
        self.model = None

    def _load_model(self):
        if self.model is None:
            set_seed()
            # Suppress verbose output from transformers if possible
            self.model = SentenceTransformer(self.model_name)

    def transform(
        self, df: pd.DataFrame, name: str, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Generates or loads embeddings for the given dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing 'text_combined'.
            name (str): Identifier for the dataset (e.g., 'train', 'test') for caching.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Dense embeddings.
        """
        cache_path = os.path.join(self.cache_dir, f"{name}_semantic.npy")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                embeddings = np.load(cache_path)
                # Simple validation of shape
                if embeddings.shape[0] == len(df):
                    return embeddings
            except Exception:
                pass  # Fallback to re-computation

        # 2. Compute from scratch
        self._load_model()

        text_data = df["text_combined"].fillna("").astype(str).tolist()

        # Encode
        embeddings = self.model.encode(
            text_data, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        # 3. Save to cache
        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(cache_path, embeddings)

        return embeddings
