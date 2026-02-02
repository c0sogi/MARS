import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from library.config import Config


class FeatureFactory:
    """
    Factory class for generating modality-specific feature sets.
    Implements strict leakage prevention (fit on train, transform test)
    and robust caching mechanisms.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        # Set random seed for reproducibility
        np.random.seed(Config.RANDOM_SEED)

    def _get_cache_paths(self, prefix):
        """Helper to generate cache paths for train and test artifacts."""
        train_path = os.path.join(self.cache_dir, f"X_train_{prefix}")
        test_path = os.path.join(self.cache_dir, f"X_test_{prefix}")
        return train_path, test_path

    def _handle_debug(self, train_df, test_df, debug_size, load_cached_data):
        """
        Slices dataframes if debug_size is set.
        Forces load_cached_data to False if debugging to avoid cache pollution.
        """
        if debug_size is not None:
            print(f"DEBUG MODE: Slicing data to {debug_size} samples.")
            train_df = train_df.iloc[:debug_size].copy()
            test_df = test_df.iloc[:debug_size].copy()
            load_cached_data = False
        return train_df, test_df, load_cached_data

    def make_lexical(self, train_df, test_df, load_cached_data=True, debug_size=None):
        """
        Generates Open-Vocabulary Lexical Features (TF-IDF).
        Granular tokenization, no max_features constraint.
        """
        train_df, test_df, load_cached_data = self._handle_debug(
            train_df, test_df, debug_size, load_cached_data
        )

        train_path, test_path = self._get_cache_paths("lexical.npz")

        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(test_path)
        ):
            print("Loading cached Lexical features...")
            X_train = sparse.load_npz(train_path)
            X_test = sparse.load_npz(test_path)
            return X_train, X_test

        print("Generating Lexical features (Open Vocabulary)...")
        # Ensure text is string
        train_text = train_df["text_combined"].astype(str).fillna("")
        test_text = test_df["text_combined"].astype(str).fillna("")

        # Fit on Train ONLY
        vectorizer = TfidfVectorizer(dtype=np.float32, **Config.TEXT_TFIDF_PARAMS)
        X_train = vectorizer.fit_transform(train_text)
        X_test = vectorizer.transform(test_text)

        if not debug_size:
            print(f"Caching Lexical features to {self.cache_dir}...")
            sparse.save_npz(train_path, X_train)
            sparse.save_npz(test_path, X_test)

        return X_train, X_test

    def make_behavioral(
        self, train_df, test_df, load_cached_data=True, debug_size=None
    ):
        """
        Generates Closed-Vocabulary Behavioral Features (TF-IDF on History).
        Constrained vocabulary to prevent overfitting.
        """
        train_df, test_df, load_cached_data = self._handle_debug(
            train_df, test_df, debug_size, load_cached_data
        )

        train_path, test_path = self._get_cache_paths("community.npz")

        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(test_path)
        ):
            print("Loading cached Behavioral features...")
            X_train = sparse.load_npz(train_path)
            X_test = sparse.load_npz(test_path)
            return X_train, X_test

        print("Generating Behavioral features (Closed Vocabulary)...")
        # Ensure history is string
        train_hist = train_df["history_str"].astype(str).fillna("")
        test_hist = test_df["history_str"].astype(str).fillna("")

        # Fit on Train ONLY
        vectorizer = TfidfVectorizer(dtype=np.float32, **Config.HISTORY_TFIDF_PARAMS)
        X_train = vectorizer.fit_transform(train_hist)
        X_test = vectorizer.transform(test_hist)

        if not debug_size:
            print(f"Caching Behavioral features to {self.cache_dir}...")
            sparse.save_npz(train_path, X_train)
            sparse.save_npz(test_path, X_test)

        return X_train, X_test

    def make_semantic(self, train_df, test_df, load_cached_data=True, debug_size=None):
        """
        Generates Dense Semantic Features using SentenceTransformers.
        Frozen embeddings (all-MiniLM-L6-v2).
        """
        train_df, test_df, load_cached_data = self._handle_debug(
            train_df, test_df, debug_size, load_cached_data
        )

        train_path, test_path = self._get_cache_paths("semantic.npy")

        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(test_path)
        ):
            print("Loading cached Semantic features...")
            X_train = np.load(train_path)
            X_test = np.load(test_path)
            return X_train, X_test

        print("Generating Semantic features (Dense Embeddings)...")
        model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Encode
        # show_progress_bar=False to reduce clutter
        X_train = model.encode(
            train_df["text_combined"].tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        X_test = model.encode(
            test_df["text_combined"].tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if not debug_size:
            print(f"Caching Semantic features to {self.cache_dir}...")
            np.save(train_path, X_train)
            np.save(test_path, X_test)

        return X_train, X_test

    def make_metadata(self, train_df, test_df, load_cached_data=True, debug_size=None):
        """
        Generates Scaled Metadata Features.
        Uses allow-listed columns and StandardScaler.
        """
        train_df, test_df, load_cached_data = self._handle_debug(
            train_df, test_df, debug_size, load_cached_data
        )

        train_path, test_path = self._get_cache_paths("meta.npy")

        if (
            load_cached_data
            and os.path.exists(train_path)
            and os.path.exists(test_path)
        ):
            print("Loading cached Metadata features...")
            X_train = np.load(train_path)
            X_test = np.load(test_path)
            return X_train, X_test

        print("Generating Metadata features (Scaled)...")
        # Extract columns
        # FillNa with 0 is already done in data_loader, but safety check here
        X_train_raw = train_df[Config.METADATA_COLS].fillna(0).values
        X_test_raw = test_df[Config.METADATA_COLS].fillna(0).values

        # Fit Scaler on Train ONLY
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)

        if not debug_size:
            print(f"Caching Metadata features to {self.cache_dir}...")
            np.save(train_path, X_train)
            np.save(test_path, X_test)

        return X_train, X_test
