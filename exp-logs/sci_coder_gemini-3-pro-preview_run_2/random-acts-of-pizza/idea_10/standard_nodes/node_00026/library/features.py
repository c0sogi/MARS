import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import QuantileTransformer, normalize
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import load_dataset

# Initialize Logger
logger = setup_logger("feature_engineering")


class HybridFeaturePipeline:
    """
    Implements the Hybrid Lexical-Semantic 'Wide-and-Deep' feature engineering pipeline.
    Combines dense semantic embeddings, sparse lexical features, community profiles,
    and rank-normalized tabular metadata.
    """

    def __init__(self):
        # 1. Semantic View
        self.sentence_model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)

        # 2. Lexical View
        self.tfidf_text = TfidfVectorizer(**Config.TFIDF_TEXT_PARAMS)

        # 3. Community View
        self.tfidf_community = TfidfVectorizer(**Config.TFIDF_SUBREDDIT_PARAMS)

        # 4. Tabular View
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )

        # State tracking
        self.is_fitted = False

    def _prepare_text(self, df):
        """Concatenates title and body text."""
        # Fill NaNs with empty strings and concatenate
        title = df[Config.TEXT_COLS[0]].fillna("").astype(str)
        body = df[Config.TEXT_COLS[1]].fillna("").astype(str)
        return (title + " " + body).tolist()

    def _prepare_subreddits(self, df):
        """Converts list of subreddits to space-separated strings."""

        # Handle cases where the column might be lists or strings (if loaded from csv vs json)
        # Assuming data_loader preserves lists from JSON or we need to parse if stringified
        def process_entry(entry):
            if isinstance(entry, list):
                return " ".join(entry)
            elif isinstance(entry, str):
                # If it looks like a list representation "[a, b]", clean it
                # But data_loader from JSON usually keeps it as list.
                # Fallback for safety if stringified:
                return (
                    entry.replace("[", "")
                    .replace("]", "")
                    .replace("'", "")
                    .replace(",", "")
                )
            return ""

        return df[Config.SUBREDDIT_COL].apply(process_entry).tolist()

    def fit(self, df):
        """Fits the stateful transformers on the training data."""
        logger.info("Fitting feature pipeline...")

        # Fit Lexical TF-IDF
        text_data = self._prepare_text(df)
        self.tfidf_text.fit(text_data)

        # Fit Community TF-IDF
        subreddit_data = self._prepare_subreddits(df)
        self.tfidf_community.fit(subreddit_data)

        # Fit Tabular Scalers
        numeric_data = df[Config.NUMERIC_COLS].values
        self.imputer.fit(numeric_data)
        imputed_data = self.imputer.transform(numeric_data)
        self.scaler.fit(imputed_data)

        self.is_fitted = True
        return self

    def transform(self, df):
        """Transforms the dataframe into the concatenated feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform.")

        logger.info("Transforming data...")

        # 1. Semantic Features (Dense)
        text_data = self._prepare_text(df)
        # Encode returns numpy array
        semantic_embeddings = self.sentence_model.encode(
            text_data, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        # L2 Normalize
        semantic_embeddings = normalize(semantic_embeddings, norm="l2")

        # 2. Lexical Features (Sparse -> Dense)
        lexical_features = self.tfidf_text.transform(text_data)
        # Convert to dense for concatenation (dimensionality is constrained by max_features)
        lexical_features = lexical_features.toarray()
        # TF-IDF vectorizer usually applies norm='l2' by default or via config,
        # but we ensure it based on Config params.

        # 3. Community Features (Sparse -> Dense)
        subreddit_data = self._prepare_subreddits(df)
        community_features = self.tfidf_community.transform(subreddit_data)
        community_features = community_features.toarray()

        # 4. Tabular Features
        numeric_data = df[Config.NUMERIC_COLS].values
        numeric_data = self.imputer.transform(numeric_data)
        tabular_features = self.scaler.transform(numeric_data)

        # 5. Fusion
        # Concatenate all features
        X_combined = np.hstack(
            [
                semantic_embeddings,
                lexical_features,
                community_features,
                tabular_features,
            ]
        )

        return X_combined.astype(np.float32)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def generate_features(load_cached_data=True):
    """
    Orchestrates the feature generation process.
    Loads data, checks cache, generates features if needed, and saves them.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    set_seed()

    # Define cache file paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(
            cache_dir, "test_ids.npy"
        ),  # Save IDs to align predictions
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and all_cached:
        logger.info("Loading features from cache...")
        try:
            X_train = np.load(paths["X_train"])
            y_train = np.load(paths["y_train"])
            X_val = np.load(paths["X_val"])
            y_val = np.load(paths["y_val"])
            X_test = np.load(paths["X_test"])
            test_ids = np.load(paths["test_ids"], allow_pickle=True)
            return X_train, y_train, X_val, y_val, X_test, test_ids
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Regenerating features.")

    # If not cached or failed, generate from scratch
    logger.info("Generating features from scratch...")

    # Load DataFrames
    df_train, df_val, df_test = load_dataset(load_cached_data=load_cached_data)

    # Initialize Pipeline
    pipeline = HybridFeaturePipeline()

    # Fit on Train
    logger.info("Fitting pipeline on training set...")
    pipeline.fit(df_train)

    # Transform all splits
    logger.info("Transforming training set...")
    X_train = pipeline.transform(df_train)
    y_train = df_train[Config.TARGET_COL].values.astype(int)

    logger.info("Transforming validation set...")
    X_val = pipeline.transform(df_val)
    y_val = df_val[Config.TARGET_COL].values.astype(int)

    logger.info("Transforming test set...")
    X_test = pipeline.transform(df_test)
    test_ids = df_test[Config.ID_COL].values

    # Save to cache
    logger.info("Saving features to cache...")
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["X_test"], X_test)
    np.save(paths["test_ids"], test_ids)

    logger.info(f"Feature generation complete. Feature shape: {X_train.shape}")

    return X_train, y_train, X_val, y_val, X_test, test_ids
