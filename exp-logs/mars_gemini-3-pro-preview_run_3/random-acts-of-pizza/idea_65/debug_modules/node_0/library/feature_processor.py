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
    METADATA_COLS,
    LEXICAL_VECTORIZER_PARAMS,
    COMMUNITY_VECTORIZER_PARAMS,
    TARGET_COL,
    RANDOM_SEED,
)
from library.utils import setup_logger

logger = setup_logger("feature_processor")


class FeaturePipeline:
    """
    Orchestrates feature engineering for the Symmetric Non-View Stacking Ensemble.
    Generates four distinct views of the data:
    1. Lexical (Sparse TF-IDF)
    2. Behavioral (Sparse Bag-of-Communities)
    3. Semantic (Dense Embeddings)
    4. Metadata (Dense Numerical)
    """

    def __init__(self):
        # Branch 1: Sparse Lexical
        self.lexical_vectorizer = TfidfVectorizer(**LEXICAL_VECTORIZER_PARAMS)

        # Branch 2: Sparse Behavioral
        self.community_vectorizer = TfidfVectorizer(**COMMUNITY_VECTORIZER_PARAMS)

        # Branch 3: Dense Semantic
        # We initialize this lazily or in fit to avoid overhead if loading from cache
        self.embedding_model = None

        # Branch 4: Contextual Metadata
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def _get_text_data(self, df):
        """Concatenates title and edit-aware body text."""
        # Fill NaNs with empty string
        title = df[TEXT_COLS[0]].fillna("").astype(str)
        body = df[TEXT_COLS[1]].fillna("").astype(str)
        return title + " " + body

    def _get_community_data(self, df):
        """Processes subreddit lists into space-separated strings for vectorization."""

        # The column contains lists of strings. We join them.
        # Handle cases where it might be NaN or empty list
        def join_subreddits(x):
            if isinstance(x, list):
                return " ".join(x)
            elif isinstance(x, str):
                return x
            return ""

        return df[SUBREDDIT_COL].apply(join_subreddits)

    def fit(self, df):
        """Fits all transformers on the training data."""
        logger.info("Fitting Lexical Vectorizer...")
        text_data = self._get_text_data(df)
        self.lexical_vectorizer.fit(text_data)

        logger.info("Fitting Community Vectorizer...")
        community_data = self._get_community_data(df)
        self.community_vectorizer.fit(community_data)

        logger.info("Fitting Metadata Scaler/Imputer...")
        # Select allowed metadata columns
        meta_data = df[METADATA_COLS].copy()
        meta_data = self.imputer.fit_transform(meta_data)
        self.scaler.fit(meta_data)

        return self

    def transform(self, df, is_train=False):
        """
        Transforms data into the 4 feature sets.

        Args:
            df (pd.DataFrame): Data to transform.
            is_train (bool): If True, returns y (target) as well.

        Returns:
            dict: Dictionary containing 'X_lexical', 'X_community', 'X_semantic', 'X_metadata'
                  and optionally 'y'.
        """
        features = {}

        # 1. Lexical Features (Sparse)
        logger.info("Transforming Lexical Features...")
        text_data = self._get_text_data(df)
        features["X_lexical"] = self.lexical_vectorizer.transform(text_data).astype(
            np.float32
        )

        # 2. Behavioral Features (Sparse)
        logger.info("Transforming Behavioral Features...")
        community_data = self._get_community_data(df)
        features["X_community"] = self.community_vectorizer.transform(
            community_data
        ).astype(np.float32)

        # 3. Semantic Features (Dense)
        logger.info("Transforming Semantic Features (Embeddings)...")
        if self.embedding_model is None:
            # Initialize model on CPU/GPU automatically
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode returns numpy array by default
        # We use a batch size to manage memory, though dataset is small enough
        embeddings = self.embedding_model.encode(
            text_data.tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        features["X_semantic"] = embeddings.astype(np.float32)

        # 4. Metadata Features (Dense)
        logger.info("Transforming Metadata Features...")
        meta_data = df[METADATA_COLS].copy()
        meta_data = self.imputer.transform(meta_data)
        features["X_metadata"] = self.scaler.transform(meta_data).astype(np.float32)

        # 5. Target
        if is_train and TARGET_COL in df.columns:
            features["y"] = df[TARGET_COL].values.astype(np.int32)

        return features


def get_processed_features(train_df, test_df, load_cached_data=True):
    """
    Main entry point for feature processing with caching mechanism.

    Args:
        train_df (pd.DataFrame): Training data (Union of Train+Val).
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_features_dict, test_features_dict)
    """
    # Define cache filenames
    filenames = {
        "train": {
            "X_lexical": "X_train_lexical.npz",
            "X_community": "X_train_community.npz",
            "X_semantic": "X_train_semantic.npy",
            "X_metadata": "X_train_metadata.npy",
            "y": "y_train.npy",
        },
        "test": {
            "X_lexical": "X_test_lexical.npz",
            "X_community": "X_test_community.npz",
            "X_semantic": "X_test_semantic.npy",
            "X_metadata": "X_test_metadata.npy",
        },
    }

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check if all files exist
    all_files_exist = True
    if load_cached_data:
        for split in filenames:
            for key, fname in filenames[split].items():
                if not os.path.exists(os.path.join(CACHE_DIR, fname)):
                    all_files_exist = False
                    break

    if load_cached_data and all_files_exist:
        logger.info(f"Loading processed features from cache: {CACHE_DIR}")
        try:
            train_feats = {}
            test_feats = {}

            # Load Train
            train_feats["X_lexical"] = sp.load_npz(
                os.path.join(CACHE_DIR, filenames["train"]["X_lexical"])
            )
            train_feats["X_community"] = sp.load_npz(
                os.path.join(CACHE_DIR, filenames["train"]["X_community"])
            )
            train_feats["X_semantic"] = np.load(
                os.path.join(CACHE_DIR, filenames["train"]["X_semantic"])
            )
            train_feats["X_metadata"] = np.load(
                os.path.join(CACHE_DIR, filenames["train"]["X_metadata"])
            )
            train_feats["y"] = np.load(os.path.join(CACHE_DIR, filenames["train"]["y"]))

            # Load Test
            test_feats["X_lexical"] = sp.load_npz(
                os.path.join(CACHE_DIR, filenames["test"]["X_lexical"])
            )
            test_feats["X_community"] = sp.load_npz(
                os.path.join(CACHE_DIR, filenames["test"]["X_community"])
            )
            test_feats["X_semantic"] = np.load(
                os.path.join(CACHE_DIR, filenames["test"]["X_semantic"])
            )
            test_feats["X_metadata"] = np.load(
                os.path.join(CACHE_DIR, filenames["test"]["X_metadata"])
            )

            logger.info("Successfully loaded features from cache.")
            return train_feats, test_feats

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-processing...")

    # Process from scratch
    logger.info("Processing features from scratch...")
    pipeline = FeaturePipeline()

    # Fit on Train
    pipeline.fit(train_df)

    # Transform Train and Test
    train_feats = pipeline.transform(train_df, is_train=True)
    test_feats = pipeline.transform(test_df, is_train=False)

    # Save to Cache
    logger.info(f"Saving features to cache: {CACHE_DIR}")

    # Save Train
    sp.save_npz(
        os.path.join(CACHE_DIR, filenames["train"]["X_lexical"]),
        train_feats["X_lexical"],
    )
    sp.save_npz(
        os.path.join(CACHE_DIR, filenames["train"]["X_community"]),
        train_feats["X_community"],
    )
    np.save(
        os.path.join(CACHE_DIR, filenames["train"]["X_semantic"]),
        train_feats["X_semantic"],
    )
    np.save(
        os.path.join(CACHE_DIR, filenames["train"]["X_metadata"]),
        train_feats["X_metadata"],
    )
    np.save(os.path.join(CACHE_DIR, filenames["train"]["y"]), train_feats["y"])

    # Save Test
    sp.save_npz(
        os.path.join(CACHE_DIR, filenames["test"]["X_lexical"]), test_feats["X_lexical"]
    )
    sp.save_npz(
        os.path.join(CACHE_DIR, filenames["test"]["X_community"]),
        test_feats["X_community"],
    )
    np.save(
        os.path.join(CACHE_DIR, filenames["test"]["X_semantic"]),
        test_feats["X_semantic"],
    )
    np.save(
        os.path.join(CACHE_DIR, filenames["test"]["X_metadata"]),
        test_feats["X_metadata"],
    )

    logger.info("Feature processing and caching complete.")

    return train_feats, test_feats
