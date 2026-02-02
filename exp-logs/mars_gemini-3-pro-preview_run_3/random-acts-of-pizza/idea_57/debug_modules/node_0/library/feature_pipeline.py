import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import setup_logger, set_seed

# Initialize Logger
logger = setup_logger("feature_pipeline")


class FeatureManager:
    """
    Manages the extraction, transformation, and caching of features for the stacking ensemble.
    Handles four distinct modalities:
    1. Lexical: Sparse TF-IDF of text content.
    2. Behavioral: Sparse TF-IDF of subreddit history.
    3. Semantic: Dense embeddings using a pre-trained transformer.
    4. Metadata: Dense vector of user statistics and timestamps.
    """

    def __init__(self):
        set_seed(Config.RANDOM_STATE)

        # Initialize Transformers based on Config
        self.lexical_vectorizer = TfidfVectorizer(**Config.TEXT_TFIDF_PARAMS)
        self.community_vectorizer = TfidfVectorizer(**Config.SUBREDDIT_TFIDF_PARAMS)

        # Metadata transformers
        self.meta_imputer = SimpleImputer(strategy="median")
        self.meta_scaler = StandardScaler()

        # Semantic transformers
        self.semantic_scaler = StandardScaler()
        # Embedding model is loaded lazily or in the method to save resources if cached

    def _get_cache_paths(self):
        """Returns a dictionary of file paths for caching."""
        return {
            "train_lexical": os.path.join(Config.CACHE_DIR, "X_train_lexical.npz"),
            "test_lexical": os.path.join(Config.CACHE_DIR, "X_test_lexical.npz"),
            "train_community": os.path.join(Config.CACHE_DIR, "X_train_community.npz"),
            "test_community": os.path.join(Config.CACHE_DIR, "X_test_community.npz"),
            "train_semantic": os.path.join(Config.CACHE_DIR, "X_train_semantic.npy"),
            "test_semantic": os.path.join(Config.CACHE_DIR, "X_test_semantic.npy"),
            "train_meta": os.path.join(Config.CACHE_DIR, "X_train_meta.npy"),
            "test_meta": os.path.join(Config.CACHE_DIR, "X_test_meta.npy"),
        }

    def _check_cache_exists(self, paths):
        """Checks if all cache files exist."""
        return all(os.path.exists(p) for p in paths.values())

    def _load_cache(self, paths):
        """Loads features from cache."""
        logger.info("Loading features from cache...")
        data = {}
        data["lexical_sparse"] = (
            sp.load_npz(paths["train_lexical"]),
            sp.load_npz(paths["test_lexical"]),
        )
        data["community_sparse"] = (
            sp.load_npz(paths["train_community"]),
            sp.load_npz(paths["test_community"]),
        )
        data["semantic_dense"] = (
            np.load(paths["train_semantic"]),
            np.load(paths["test_semantic"]),
        )
        data["metadata_only"] = (
            np.load(paths["train_meta"]),
            np.load(paths["test_meta"]),
        )
        return data

    def _save_cache(self, data, paths):
        """Saves computed features to cache."""
        logger.info(f"Saving features to cache directory: {Config.CACHE_DIR}")

        # Lexical
        sp.save_npz(paths["train_lexical"], data["lexical_sparse"][0])
        sp.save_npz(paths["test_lexical"], data["lexical_sparse"][1])

        # Community
        sp.save_npz(paths["train_community"], data["community_sparse"][0])
        sp.save_npz(paths["test_community"], data["community_sparse"][1])

        # Semantic
        np.save(paths["train_semantic"], data["semantic_dense"][0])
        np.save(paths["test_semantic"], data["semantic_dense"][1])

        # Metadata
        np.save(paths["train_meta"], data["metadata_only"][0])
        np.save(paths["test_meta"], data["metadata_only"][1])

    def _prepare_text(self, df):
        """Concatenates title and body for text processing."""
        # Fill NaNs with empty string just in case
        title = df["request_title"].fillna("").astype(str)
        body = df["request_text_edit_aware"].fillna("").astype(str)
        return title + " " + body

    def _prepare_subreddits(self, df):
        """Joins the list of subreddits into a space-separated string."""

        # Handle cases where the column might be NaN or empty lists
        def join_subs(x):
            if isinstance(x, list):
                return " ".join(x)
            elif isinstance(x, np.ndarray):
                return " ".join(x)
            return ""

        return df[Config.SUBREDDIT_COL].apply(join_subs)

    def process_features(self, train_df, test_df, load_cached_data=True):
        """
        Main method to generate all feature sets.

        Args:
            train_df (pd.DataFrame): The union training dataset.
            test_df (pd.DataFrame): The test dataset.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: A dictionary where keys are feature set names (e.g., 'lexical_sparse')
                  and values are tuples (X_train, X_test).
        """
        paths = self._get_cache_paths()

        # 1. Try Loading from Cache
        if load_cached_data and self._check_cache_exists(paths):
            try:
                return self._load_cache(paths)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing features...")

        logger.info("Computing features from scratch...")

        # --- Preprocessing ---
        logger.info("Preprocessing text and behavioral data...")
        train_text = self._prepare_text(train_df)
        test_text = self._prepare_text(test_df)

        train_subs = self._prepare_subreddits(train_df)
        test_subs = self._prepare_subreddits(test_df)

        # --- 1. Lexical Features (Sparse TF-IDF) ---
        logger.info("Generating Lexical Features (TF-IDF)...")
        X_train_lex = self.lexical_vectorizer.fit_transform(train_text)
        X_test_lex = self.lexical_vectorizer.transform(test_text)

        # --- 2. Behavioral Features (Sparse Community TF-IDF) ---
        logger.info("Generating Behavioral Features (Subreddit History)...")
        X_train_comm = self.community_vectorizer.fit_transform(train_subs)
        X_test_comm = self.community_vectorizer.transform(test_subs)

        # --- 3. Semantic Features (Dense Embeddings) ---
        logger.info(f"Generating Semantic Features ({Config.EMBEDDING_MODEL})...")
        # Load model here to avoid memory usage if cache was hit
        embedder = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Encode
        # Note: SentenceTransformer handles batching internally usually, but for this dataset size
        # passing the list/series directly is efficient enough.
        X_train_sem_raw = embedder.encode(train_text.tolist(), show_progress_bar=False)
        X_test_sem_raw = embedder.encode(test_text.tolist(), show_progress_bar=False)

        # Scale
        X_train_sem = self.semantic_scaler.fit_transform(X_train_sem_raw)
        X_test_sem = self.semantic_scaler.transform(X_test_sem_raw)

        # --- 4. Metadata Features (Dense) ---
        logger.info("Generating Metadata Features...")
        # Select columns
        meta_cols = Config.METADATA_COLS

        # Extract raw numpy arrays
        X_train_meta_raw = train_df[meta_cols].values
        X_test_meta_raw = test_df[meta_cols].values

        # Impute
        X_train_meta_imp = self.meta_imputer.fit_transform(X_train_meta_raw)
        X_test_meta_imp = self.meta_imputer.transform(X_test_meta_raw)

        # Scale
        X_train_meta = self.meta_scaler.fit_transform(X_train_meta_imp)
        X_test_meta = self.meta_scaler.transform(X_test_meta_imp)

        # --- Construct Result Dictionary ---
        data = {
            "lexical_sparse": (X_train_lex, X_test_lex),
            "community_sparse": (X_train_comm, X_test_comm),
            "semantic_dense": (X_train_sem, X_test_sem),
            "metadata_only": (X_train_meta, X_test_meta),
        }

        # --- Save to Cache ---
        self._save_cache(data, paths)

        return data
