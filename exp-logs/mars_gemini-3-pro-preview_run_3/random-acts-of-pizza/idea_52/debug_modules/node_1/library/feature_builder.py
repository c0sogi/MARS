import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import (
    CACHE_DIR,
    TEXT_COLS,
    SUBREDDIT_COL,
    NUMERICAL_COLS,
    TFIDF_PARAMS,
    COMMUNITY_PARAMS,
    EMBEDDING_MODEL_NAME,
    SEED,
)
from library.utils import set_seed, get_device, log_metric


class FeaturePipeline:
    def __init__(self, cache_dir=CACHE_DIR):
        """
        Initializes the FeaturePipeline with necessary configurations and
        placeholder objects for transformers.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Lexical Branch (Text)
        self.lexical_vectorizer = TfidfVectorizer(**TFIDF_PARAMS)

        # Behavioral Branch (Subreddits)
        self.community_vectorizer = TfidfVectorizer(**COMMUNITY_PARAMS)

        # Contextual Branch (Metadata)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # Semantic Branch (Embeddings) - Model loaded on demand or during transform
        self.embedding_model_name = EMBEDDING_MODEL_NAME

        set_seed(SEED)

    def _prepare_text(self, df):
        """Concatenates title and edit-aware body text."""
        # Fill NA with empty string to avoid errors
        title = df[TEXT_COLS[0]].fillna("").astype(str)
        body = df[TEXT_COLS[1]].fillna("").astype(str)
        return title + " " + body

    def _prepare_subreddits(self, df):
        """Converts list of subreddits to space-separated string."""

        def join_subreddits(x):
            if isinstance(x, (list, np.ndarray)):
                return " ".join(x)
            return ""

        return df[SUBREDDIT_COL].apply(join_subreddits)

    def fit(self, df_train):
        """
        Fits the vectorizers and scalers on the training data.

        Args:
            df_train (pd.DataFrame): The training dataset.
        """
        print("Fitting FeaturePipeline on training data...")

        # 1. Fit Lexical Vectorizer
        text_data = self._prepare_text(df_train)
        self.lexical_vectorizer.fit(text_data)

        # 2. Fit Behavioral Vectorizer
        community_data = self._prepare_subreddits(df_train)
        self.community_vectorizer.fit(community_data)

        # 3. Fit Metadata Preprocessors
        # Ensure we only use the columns present in the config
        meta_data = df_train[NUMERICAL_COLS].copy()
        # Handle potential non-numeric coercion if dirty data exists, though parquet is typed
        meta_data = meta_data.apply(pd.to_numeric, errors="coerce")

        self.imputer.fit(meta_data)
        imputed_meta = self.imputer.transform(meta_data)
        self.scaler.fit(imputed_meta)

        print("FeaturePipeline fitting complete.")
        return self

    def transform(self, df, split_name, load_cached_data=True):
        """
        Transforms the dataset into feature matrices. Checks cache first.

        Args:
            df (pd.DataFrame): The dataset to transform.
            split_name (str): 'train', 'val', or 'test' (used for cache keys).
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Dictionary containing 'X_lexical', 'X_behavioral', 'X_semantic', 'X_meta'.
        """
        # Define cache paths
        paths = {
            "X_lexical": os.path.join(self.cache_dir, f"X_{split_name}_lexical.npz"),
            "X_behavioral": os.path.join(
                self.cache_dir, f"X_{split_name}_behavioral.npz"
            ),
            "X_semantic": os.path.join(self.cache_dir, f"X_{split_name}_semantic.npy"),
            "X_meta": os.path.join(self.cache_dir, f"X_{split_name}_meta.npy"),
        }

        # Check if all cache files exist
        cache_exists = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and cache_exists:
            print(f"Loading cached features for split: {split_name}")
            features = {}
            features["X_lexical"] = sp.load_npz(paths["X_lexical"])
            features["X_behavioral"] = sp.load_npz(paths["X_behavioral"])
            features["X_semantic"] = np.load(paths["X_semantic"])
            features["X_meta"] = np.load(paths["X_meta"])
            return features

        print(f"Computing features for split: {split_name}...")

        # 1. Lexical Features (Sparse)
        text_data = self._prepare_text(df)
        X_lexical = self.lexical_vectorizer.transform(text_data)
        sp.save_npz(paths["X_lexical"], X_lexical)

        # 2. Behavioral Features (Sparse)
        community_data = self._prepare_subreddits(df)
        X_behavioral = self.community_vectorizer.transform(community_data)
        sp.save_npz(paths["X_behavioral"], X_behavioral)

        # 3. Semantic Features (Dense)
        # Load model only when needed to save memory
        device = get_device()
        print(f"Generating embeddings using {self.embedding_model_name} on {device}...")
        model = SentenceTransformer(self.embedding_model_name, device=str(device))

        # SentenceTransformer handles batching internally, but we pass list of strings
        # Converting Series to list
        text_list = text_data.tolist()
        X_semantic = model.encode(
            text_list, show_progress_bar=False, convert_to_numpy=True
        )

        # Free up GPU memory
        del model
        if device.type == "cuda":
            import torch

            torch.cuda.empty_cache()

        np.save(paths["X_semantic"], X_semantic)

        # 4. Metadata Features (Dense)
        meta_data = df[NUMERICAL_COLS].copy()
        meta_data = meta_data.apply(pd.to_numeric, errors="coerce")
        X_meta = self.imputer.transform(meta_data)
        X_meta = self.scaler.transform(X_meta)
        np.save(paths["X_meta"], X_meta)

        return {
            "X_lexical": X_lexical,
            "X_behavioral": X_behavioral,
            "X_semantic": X_semantic,
            "X_meta": X_meta,
        }
