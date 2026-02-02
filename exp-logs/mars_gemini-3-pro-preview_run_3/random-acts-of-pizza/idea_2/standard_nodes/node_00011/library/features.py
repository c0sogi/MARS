import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("feature_processor")


class FeatureProcessor:
    """
    Handles feature engineering, including text processing, semantic embeddings,
    and metadata transformation.
    """

    def __init__(self):
        # Lexical-Linear Branch: TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
        )

        # Dense Metadata Processing
        self.imputer = SimpleImputer(strategy="median")
        self.scaler_dense = StandardScaler()

        # Semantic Branch: Embeddings
        self.scaler_embed = StandardScaler()
        self.embed_model = None  # Lazy initialization

    def _load_embedding_model(self):
        """Loads the SentenceTransformer model if not already loaded."""
        if self.embed_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading SentenceTransformer model on {device}...")
            self.embed_model = SentenceTransformer(
                Config.TRANSFORMER_MODEL, device=device
            )

    def _generate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates derived numerical features from timestamps and text."""
        df_derived = pd.DataFrame(index=df.index)

        # Time-based features
        # Convert timestamp to datetime objects
        dt_series = pd.to_datetime(df["unix_timestamp_of_request_utc"], unit="s")
        df_derived["request_hour"] = dt_series.dt.hour
        df_derived["request_day_of_week"] = dt_series.dt.dayofweek

        # Text length features
        # Fill NaNs with empty string for safety
        text_col = df["request_text_edit_aware"].fillna("").astype(str)
        title_col = df["request_title"].fillna("").astype(str)

        df_derived["text_word_count"] = text_col.apply(lambda x: len(x.split()))
        df_derived["title_word_count"] = title_col.apply(lambda x: len(x.split()))

        return df_derived

    def _prepare_text_corpus(self, df: pd.DataFrame) -> list:
        """
        Combines title, text, and flattened subreddits into a single string per sample.
        """
        # 1. Title
        titles = df["request_title"].fillna("").astype(str)

        # 2. Request Text (Edit Aware)
        texts = df["request_text_edit_aware"].fillna("").astype(str)

        # 3. Subreddits (Flatten list to string)
        # Handle cases where the column might be NaN or not a list
        def flatten_subreddits(x):
            if isinstance(x, list):
                return " ".join(x)
            return ""

        subreddits = df[Config.SUBREDDIT_COL].apply(flatten_subreddits)

        # Concatenate with spaces
        combined = titles + " " + texts + " " + subreddits
        return combined.tolist()

    def _prepare_dense_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selects raw numerical columns and concatenates with derived features.
        """
        # Select raw numerical columns defined in Config
        # Ensure they exist in df (some might be missing in test if not careful, but schema guarantees them)
        raw_dense = df[Config.NUMERICAL_COLS].copy()

        # Generate derived features
        derived_dense = self._generate_derived_features(df)

        # Concatenate
        dense_df = pd.concat([raw_dense, derived_dense], axis=1)
        return dense_df

    def fit(self, df: pd.DataFrame):
        """
        Fits the internal transformers (Imputer, Scalers, TF-IDF) on the training data.
        """
        logger.info("Fitting FeatureProcessor...")

        # 1. Fit Dense Pipeline
        dense_df = self._prepare_dense_features(df)
        self.imputer.fit(dense_df)
        dense_imputed = self.imputer.transform(dense_df)
        self.scaler_dense.fit(dense_imputed)

        # 2. Fit TF-IDF
        text_corpus = self._prepare_text_corpus(df)
        self.tfidf.fit(text_corpus)

        # 3. Fit Embedding Scaler
        # We need to generate embeddings for the train set to fit the scaler
        self._load_embedding_model()
        logger.info("Generating embeddings for scaler fitting...")
        embeddings = self.embed_model.encode(
            text_corpus,
            batch_size=Config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.scaler_embed.fit(embeddings)

        logger.info("FeatureProcessor fitted successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> dict:
        """
        Transforms the data into feature dictionaries.
        Returns:
            dict: {'tfidf': sparse_matrix, 'embedding': np.array, 'dense': np.array}
        """
        # 1. Process Dense Features
        dense_df = self._prepare_dense_features(df)
        dense_imputed = self.imputer.transform(dense_df)
        dense_scaled = self.scaler_dense.transform(dense_imputed)

        # 2. Process TF-IDF
        text_corpus = self._prepare_text_corpus(df)
        tfidf_matrix = self.tfidf.transform(text_corpus)

        # 3. Process Embeddings
        self._load_embedding_model()
        embeddings = self.embed_model.encode(
            text_corpus,
            batch_size=Config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings_scaled = self.scaler_embed.transform(embeddings)

        return {
            "tfidf": tfidf_matrix,
            "embedding": embeddings_scaled,
            "dense": dense_scaled,
        }


def process_and_cache_data(train_df, val_df, test_df, load_cached_data=True):
    """
    Orchestrates the feature processing pipeline with caching.

    Args:
        train_df, val_df, test_df: Pandas DataFrames.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: ((X_train, y_train), (X_val, y_val), (X_test, None))
               where X is a dictionary of features.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define filenames
    files = {
        "train": {
            "tfidf": os.path.join(cache_dir, "X_train_tfidf.npz"),
            "embedding": os.path.join(cache_dir, "X_train_embed.npy"),
            "dense": os.path.join(cache_dir, "X_train_dense.npy"),
            "y": os.path.join(cache_dir, "y_train.npy"),
        },
        "val": {
            "tfidf": os.path.join(cache_dir, "X_val_tfidf.npz"),
            "embedding": os.path.join(cache_dir, "X_val_embed.npy"),
            "dense": os.path.join(cache_dir, "X_val_dense.npy"),
            "y": os.path.join(cache_dir, "y_val.npy"),
        },
        "test": {
            "tfidf": os.path.join(cache_dir, "X_test_tfidf.npz"),
            "embedding": os.path.join(cache_dir, "X_test_embed.npy"),
            "dense": os.path.join(cache_dir, "X_test_dense.npy"),
            # No target for test
        },
    }

    # Helper to check existence
    def check_cache_exists():
        for split, paths in files.items():
            for key, path in paths.items():
                if not os.path.exists(path):
                    return False
        return True

    # Helper to load
    def load_cache():
        logger.info("Loading features from cache...")
        data = {}
        for split in ["train", "val", "test"]:
            X = {}
            X["tfidf"] = sp.load_npz(files[split]["tfidf"])
            X["embedding"] = np.load(files[split]["embedding"])
            X["dense"] = np.load(files[split]["dense"])

            if "y" in files[split]:
                y = np.load(files[split]["y"])
            else:
                y = None
            data[split] = (X, y)
        return data["train"], data["val"], data["test"]

    # Helper to save
    def save_cache(X, y, split):
        logger.info(f"Saving {split} features to cache...")
        sp.save_npz(files[split]["tfidf"], X["tfidf"])
        np.save(files[split]["embedding"], X["embedding"])
        np.save(files[split]["dense"], X["dense"])
        if y is not None:
            np.save(files[split]["y"], y)

    # Execution Logic
    if load_cached_data and check_cache_exists():
        try:
            return load_cache()
        except Exception as e:
            logger.warning(f"Failed to load cache ({e}). Re-computing...")

    # Compute from scratch
    logger.info("Computing features from scratch...")
    processor = FeatureProcessor()

    # Fit on Train
    processor.fit(train_df)

    # Transform all splits
    logger.info("Transforming Train set...")
    X_train = processor.transform(train_df)
    y_train = train_df[Config.TARGET_COL].values.astype(int)

    logger.info("Transforming Validation set...")
    X_val = processor.transform(val_df)
    y_val = val_df[Config.TARGET_COL].values.astype(int)

    logger.info("Transforming Test set...")
    X_test = processor.transform(test_df)
    y_test = None  # Test set has no target

    # Save to cache
    save_cache(X_train, y_train, "train")
    save_cache(X_val, y_val, "val")
    save_cache(X_test, y_test, "test")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
