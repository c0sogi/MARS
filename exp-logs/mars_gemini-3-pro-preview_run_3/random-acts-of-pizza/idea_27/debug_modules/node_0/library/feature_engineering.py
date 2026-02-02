import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.data_loader import load_data


class FeatureFactory:
    """
    Orchestrates the creation of feature views for the Pent-View architecture.
    Manages stateful transformers (fitting on train, transforming val/test).
    """

    def __init__(self):
        # Metadata transformers
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # Lexical transformer (Text)
        self.lexical_vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)

        # Behavioral transformer (History)
        # Modified params for subreddits: unigrams only, no stop words (to keep subreddits like 'it', 'all')
        hist_params = Config.TFIDF_PARAMS.copy()
        hist_params["ngram_range"] = (1, 1)
        hist_params["stop_words"] = None
        self.behavioral_vectorizer = TfidfVectorizer(**hist_params)

        # Semantic model (Lazy loaded)
        self.embedding_model = None

    def _get_embedding_model(self):
        if self.embedding_model is None:
            # Suppress verbose output
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        return self.embedding_model

    def fit_metadata(self, df):
        """Fits imputer and scaler on training metadata."""
        X = df[Config.NUMERICAL_COLS].values
        X = self.imputer.fit_transform(X)
        self.scaler.fit(X)

    def transform_metadata(self, df):
        """Transforms metadata using fitted imputer and scaler."""
        X = df[Config.NUMERICAL_COLS].values
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return X.astype(np.float32)

    def fit_lexical(self, df):
        """Fits TF-IDF on training text."""
        texts = df[Config.TEXT_COL].fillna("").astype(str).tolist()
        self.lexical_vectorizer.fit(texts)

    def transform_lexical(self, df):
        """Transforms text to TF-IDF sparse matrix."""
        texts = df[Config.TEXT_COL].fillna("").astype(str).tolist()
        return self.lexical_vectorizer.transform(texts)

    def fit_behavioral(self, df):
        """Fits TF-IDF on training subreddit history."""
        # History is already serialized to space-separated string in data_loader
        history = df[Config.HISTORY_COL].fillna("").astype(str).tolist()
        self.behavioral_vectorizer.fit(history)

    def transform_behavioral(self, df):
        """Transforms history to TF-IDF sparse matrix."""
        history = df[Config.HISTORY_COL].fillna("").astype(str).tolist()
        return self.behavioral_vectorizer.transform(history)

    def create_semantic_view(self, df):
        """Generates dense embeddings using SentenceTransformer."""
        model = self._get_embedding_model()
        texts = df[Config.TEXT_COL].fillna("").astype(str).tolist()
        # Encode with progress bar disabled
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


def get_all_features(load_cached_data=True):
    """
    Main entry point to retrieve all feature views.
    Handles caching logic: checks for files, loads if present, else computes and saves.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        dict: Dictionary containing all feature matrices and targets.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define all cache file paths
    cache_files = {
        "X_train_meta": (Config.CACHE_TRAIN_META, np.load),
        "X_val_meta": (Config.CACHE_VAL_META, np.load),
        "X_test_meta": (Config.CACHE_TEST_META, np.load),
        "X_train_lexical": (Config.CACHE_TRAIN_TEXT_TFIDF, sparse.load_npz),
        "X_val_lexical": (Config.CACHE_VAL_TEXT_TFIDF, sparse.load_npz),
        "X_test_lexical": (Config.CACHE_TEST_TEXT_TFIDF, sparse.load_npz),
        "X_train_behavioral": (Config.CACHE_TRAIN_HIST_TFIDF, sparse.load_npz),
        "X_val_behavioral": (Config.CACHE_VAL_HIST_TFIDF, sparse.load_npz),
        "X_test_behavioral": (Config.CACHE_TEST_HIST_TFIDF, sparse.load_npz),
        "X_train_semantic": (Config.CACHE_TRAIN_EMBED, np.load),
        "X_val_semantic": (Config.CACHE_VAL_EMBED, np.load),
        "X_test_semantic": (Config.CACHE_TEST_EMBED, np.load),
        "y_train": (Config.CACHE_Y_TRAIN, np.load),
        "y_val": (Config.CACHE_Y_VAL, np.load),
    }

    # Check if all files exist
    all_cached = all(os.path.exists(path) for path, _ in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading features from cache...")
        data = {}
        for key, (path, loader) in cache_files.items():
            data[key] = loader(path)
        return data

    print("Computing features from scratch...")

    # Load raw data
    train_df, val_df, test_df = load_data()

    # Initialize Factory
    factory = FeatureFactory()

    # 1. Process Metadata
    print("Processing Metadata...")
    factory.fit_metadata(train_df)
    X_train_meta = factory.transform_metadata(train_df)
    X_val_meta = factory.transform_metadata(val_df)
    X_test_meta = factory.transform_metadata(test_df)

    # 2. Process Lexical (Sparse)
    print("Processing Lexical Features...")
    factory.fit_lexical(train_df)
    X_train_lexical = factory.transform_lexical(train_df)
    X_val_lexical = factory.transform_lexical(val_df)
    X_test_lexical = factory.transform_lexical(test_df)

    # 3. Process Behavioral (Sparse)
    print("Processing Behavioral Features...")
    factory.fit_behavioral(train_df)
    X_train_behavioral = factory.transform_behavioral(train_df)
    X_val_behavioral = factory.transform_behavioral(val_df)
    X_test_behavioral = factory.transform_behavioral(test_df)

    # 4. Process Semantic (Dense)
    print("Processing Semantic Features...")
    X_train_semantic = factory.create_semantic_view(train_df)
    X_val_semantic = factory.create_semantic_view(val_df)
    X_test_semantic = factory.create_semantic_view(test_df)

    # Extract Targets
    y_train = train_df[Config.TARGET_COL].values
    y_val = val_df[Config.TARGET_COL].values

    # Save to Cache
    print("Saving features to cache...")
    np.save(Config.CACHE_TRAIN_META, X_train_meta)
    np.save(Config.CACHE_VAL_META, X_val_meta)
    np.save(Config.CACHE_TEST_META, X_test_meta)

    sparse.save_npz(Config.CACHE_TRAIN_TEXT_TFIDF, X_train_lexical)
    sparse.save_npz(Config.CACHE_VAL_TEXT_TFIDF, X_val_lexical)
    sparse.save_npz(Config.CACHE_TEST_TEXT_TFIDF, X_test_lexical)

    sparse.save_npz(Config.CACHE_TRAIN_HIST_TFIDF, X_train_behavioral)
    sparse.save_npz(Config.CACHE_VAL_HIST_TFIDF, X_val_behavioral)
    sparse.save_npz(Config.CACHE_TEST_HIST_TFIDF, X_test_behavioral)

    np.save(Config.CACHE_TRAIN_EMBED, X_train_semantic)
    np.save(Config.CACHE_VAL_EMBED, X_val_semantic)
    np.save(Config.CACHE_TEST_EMBED, X_test_semantic)

    np.save(Config.CACHE_Y_TRAIN, y_train)
    np.save(Config.CACHE_Y_VAL, y_val)

    return {
        "X_train_meta": X_train_meta,
        "X_val_meta": X_val_meta,
        "X_test_meta": X_test_meta,
        "X_train_lexical": X_train_lexical,
        "X_val_lexical": X_val_lexical,
        "X_test_lexical": X_test_lexical,
        "X_train_behavioral": X_train_behavioral,
        "X_val_behavioral": X_val_behavioral,
        "X_test_behavioral": X_test_behavioral,
        "X_train_semantic": X_train_semantic,
        "X_val_semantic": X_val_semantic,
        "X_test_semantic": X_test_semantic,
        "y_train": y_train,
        "y_val": y_val,
    }
