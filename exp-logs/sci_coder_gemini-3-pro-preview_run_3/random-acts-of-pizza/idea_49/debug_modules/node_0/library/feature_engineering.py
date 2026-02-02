import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.data_loader import DataLoader


class FeaturePipeline:
    """
    Manages feature engineering for the Hex-View Stacking Ensemble.
    Generates and caches four distinct feature modalities:
    1. Lexical (Sparse TF-IDF of Title + Body)
    2. Community (Sparse TF-IDF of Subreddit History)
    3. Semantic (Dense Embeddings of Title + Body)
    4. Metadata (Dense Allow-listed User/Post Stats)
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the pipeline.

        Args:
            load_cached_data (bool): If True, attempts to load features from disk.
                                     If False, regenerates features from scratch.
        """
        self.load_cached_data = load_cached_data
        self.cache_dir = Config.CACHE_DIR

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load processed dataframes
        # We rely on DataLoader to handle the raw -> processed dataframe caching
        self.loader = DataLoader()
        self.train_df, self.val_df, self.test_df = self.loader.get_processed_data(
            load_cached_data=load_cached_data
        )

        # Extract targets and IDs
        self.y_train = self.train_df[Config.TARGET_COL].values
        self.y_val = self.val_df[Config.TARGET_COL].values
        self.test_ids = self.test_df[Config.ID_COL].values

    def _get_cache_path(self, name, ext):
        return os.path.join(self.cache_dir, f"{name}.{ext}")

    def _save_sparse(self, matrix, name):
        path = self._get_cache_path(name, "npz")
        sparse.save_npz(path, matrix)

    def _load_sparse(self, name):
        path = self._get_cache_path(name, "npz")
        if os.path.exists(path):
            return sparse.load_npz(path)
        return None

    def _save_dense(self, array, name):
        path = self._get_cache_path(name, "npy")
        np.save(path, array)

    def _load_dense(self, name):
        path = self._get_cache_path(name, "npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def get_lexical_features(self):
        """
        Generates Sparse Lexical Features (TF-IDF on Title + Body).
        """
        names = ["train_lexical", "val_lexical", "test_lexical"]

        # Try loading from cache
        if self.load_cached_data:
            loaded = [self._load_sparse(n) for n in names]
            if all(l is not None for l in loaded):
                print("Loaded Lexical features from cache.")
                return tuple(loaded)

        print("Generating Lexical features (TF-IDF)...")
        vectorizer = TfidfVectorizer(**Config.PARAMS_TFIDF_LEXICAL)

        # Fit on train, transform all
        X_train = vectorizer.fit_transform(self.train_df["text_combined"])
        X_val = vectorizer.transform(self.val_df["text_combined"])
        X_test = vectorizer.transform(self.test_df["text_combined"])

        # Save to cache
        self._save_sparse(X_train, names[0])
        self._save_sparse(X_val, names[1])
        self._save_sparse(X_test, names[2])

        return X_train, X_val, X_test

    def get_community_features(self):
        """
        Generates Sparse Community Features (TF-IDF on Subreddit History).
        """
        names = ["train_community", "val_community", "test_community"]

        if self.load_cached_data:
            loaded = [self._load_sparse(n) for n in names]
            if all(l is not None for l in loaded):
                print("Loaded Community features from cache.")
                return tuple(loaded)

        print("Generating Community features (Bag-of-Concepts)...")
        vectorizer = TfidfVectorizer(**Config.PARAMS_TFIDF_COMMUNITY)

        # Fit on train, transform all
        X_train = vectorizer.fit_transform(self.train_df["subreddit_text"])
        X_val = vectorizer.transform(self.val_df["subreddit_text"])
        X_test = vectorizer.transform(self.test_df["subreddit_text"])

        self._save_sparse(X_train, names[0])
        self._save_sparse(X_val, names[1])
        self._save_sparse(X_test, names[2])

        return X_train, X_val, X_test

    def get_semantic_features(self):
        """
        Generates Dense Semantic Features (Embeddings of Title + Body).
        """
        names = ["train_semantic", "val_semantic", "test_semantic"]

        if self.load_cached_data:
            loaded = [self._load_dense(n) for n in names]
            if all(l is not None for l in loaded):
                print("Loaded Semantic features from cache.")
                return tuple(loaded)

        print(f"Generating Semantic features (Embeddings: {Config.EMBEDDING_MODEL})...")
        model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Encode (returns numpy array by default)
        X_train = model.encode(
            self.train_df["text_combined"].tolist(), show_progress_bar=False
        )
        X_val = model.encode(
            self.val_df["text_combined"].tolist(), show_progress_bar=False
        )
        X_test = model.encode(
            self.test_df["text_combined"].tolist(), show_progress_bar=False
        )

        self._save_dense(X_train, names[0])
        self._save_dense(X_val, names[1])
        self._save_dense(X_test, names[2])

        return X_train, X_val, X_test

    def get_metadata_features(self):
        """
        Generates Dense Metadata Features (Allow-listed, Imputed, Scaled).
        """
        names = ["train_meta", "val_meta", "test_meta"]

        if self.load_cached_data:
            loaded = [self._load_dense(n) for n in names]
            if all(l is not None for l in loaded):
                print("Loaded Metadata features from cache.")
                return tuple(loaded)

        print("Generating Metadata features (Dense)...")
        cols = Config.DENSE_FEATURES

        # Extract raw values
        X_train_raw = self.train_df[cols].values
        X_val_raw = self.val_df[cols].values
        X_test_raw = self.test_df[cols].values

        # Impute missing values (Median)
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp = imputer.transform(X_val_raw)
        X_test_imp = imputer.transform(X_test_raw)

        # Scale features (StandardScaler)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_imp)
        X_val = scaler.transform(X_val_imp)
        X_test = scaler.transform(X_test_imp)

        self._save_dense(X_train, names[0])
        self._save_dense(X_val, names[1])
        self._save_dense(X_test, names[2])

        return X_train, X_val, X_test

    def get_all_features(self):
        """
        Orchestrates the generation/loading of all feature sets.

        Returns:
            dict: Dictionary containing all feature matrices and target vectors.
        """
        lexical = self.get_lexical_features()
        community = self.get_community_features()
        semantic = self.get_semantic_features()
        meta = self.get_metadata_features()

        return {
            # Lexical (Sparse)
            "X_train_lexical": lexical[0],
            "X_val_lexical": lexical[1],
            "X_test_lexical": lexical[2],
            # Community (Sparse)
            "X_train_community": community[0],
            "X_val_community": community[1],
            "X_test_community": community[2],
            # Semantic (Dense)
            "X_train_semantic": semantic[0],
            "X_val_semantic": semantic[1],
            "X_test_semantic": semantic[2],
            # Metadata (Dense)
            "X_train_meta": meta[0],
            "X_val_meta": meta[1],
            "X_test_meta": meta[2],
            # Targets and IDs
            "y_train": self.y_train,
            "y_val": self.y_val,
            "test_ids": self.test_ids,
        }
