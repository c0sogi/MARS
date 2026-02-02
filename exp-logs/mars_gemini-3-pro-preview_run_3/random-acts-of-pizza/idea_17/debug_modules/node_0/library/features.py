import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, SimpleImputer
from sentence_transformers import SentenceTransformer
import torch

from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    CACHE_DIR,
    ID_COL,
    TARGET_COL,
    TEXT_COL,
    TITLE_COL,
    SUBREDDIT_COL,
    LEAKAGE_SUFFIX,
    TRANSFORMER_MODEL_NAME,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_SUBLINEAR_TF,
    RANDOM_SEED,
)
from library.utils import time_execution, set_seed


class FeaturePipeline:
    def __init__(self, debug_sample_size=None):
        """
        Initializes the FeaturePipeline by loading raw metadata.

        Args:
            debug_sample_size (int, optional): If set, limits the dataset size for debugging.
        """
        self.debug_sample_size = debug_sample_size
        self._load_data()

    @time_execution
    def _load_data(self):
        """Loads train, val, and test data from Parquet metadata files."""
        print("Loading metadata files...")
        self.train_df = pd.read_parquet(TRAIN_DATA_PATH)
        self.val_df = pd.read_parquet(VAL_DATA_PATH)
        self.test_df = pd.read_parquet(TEST_DATA_PATH)

        if self.debug_sample_size:
            print(f"DEBUG MODE: Sampling {self.debug_sample_size} rows.")
            self.train_df = self.train_df.head(self.debug_sample_size)
            self.val_df = self.val_df.head(self.debug_sample_size)
            self.test_df = self.test_df.head(self.debug_sample_size)

        # Extract Targets
        self.y_train = self.train_df[TARGET_COL].values
        self.y_val = self.val_df[TARGET_COL].values

        # Extract IDs for submission
        self.test_ids = self.test_df[ID_COL].values

        print(
            f"Data Loaded. Train: {self.train_df.shape}, Val: {self.val_df.shape}, Test: {self.test_df.shape}"
        )

    def get_targets(self):
        """Returns the target arrays for training and validation."""
        return self.y_train, self.y_val

    def get_test_ids(self):
        """Returns the request IDs for the test set."""
        return self.test_ids

    def _get_cache_path(self, view_name, split, ext):
        """Helper to generate cache file paths."""
        filename = f"{view_name}_{split}{ext}"
        return os.path.join(CACHE_DIR, filename)

    @time_execution
    def get_metadata_view(self, load_cached_data=True):
        """
        Generates the Unified Metadata Vector (Contextual View).
        Includes User Stats, Temporal features, and Text Complexity.
        """
        view_name = "metadata"
        ext = ".parquet"

        paths = {
            "train": self._get_cache_path(view_name, "train", ext),
            "val": self._get_cache_path(view_name, "val", ext),
            "test": self._get_cache_path(view_name, "test", ext),
        }

        # Check Cache
        if load_cached_data and all(os.path.exists(p) for p in paths.values()):
            print(f"Loading cached {view_name} view...")
            return (
                pd.read_parquet(paths["train"]),
                pd.read_parquet(paths["val"]),
                pd.read_parquet(paths["test"]),
            )

        print(f"Computing {view_name} view...")

        # Feature Engineering Helper
        def extract_features(df):
            # 1. User Stats (Allow-list)
            cols = [
                "requester_account_age_in_days_at_request",
                "requester_upvotes_minus_downvotes_at_request",
                "requester_number_of_comments_at_request",
                "requester_number_of_posts_at_request",
                "requester_number_of_subreddits_at_request",
                "requester_days_since_first_post_on_raop_at_request",
            ]
            # Ensure columns exist (fill with 0 if missing in raw, though metadata should have them)
            meta = df[cols].copy().fillna(0)

            # 2. Temporal Features
            # Convert timestamp to datetime
            dt = pd.to_datetime(df["unix_timestamp_of_request_utc"], unit="s")
            meta["request_hour"] = dt.dt.hour
            meta["request_day_of_week"] = dt.dt.dayofweek

            # 3. Text Complexity
            # Fill NA text with empty string
            text = df[TEXT_COL].fillna("").astype(str)
            meta["text_len_char"] = text.apply(len)
            meta["text_len_word"] = text.apply(lambda x: len(x.split()))

            return meta

        X_train_raw = extract_features(self.train_df)
        X_val_raw = extract_features(self.val_df)
        X_test_raw = extract_features(self.test_df)

        # Imputation
        imputer = SimpleImputer(strategy="median")
        cols = X_train_raw.columns

        X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=cols)
        X_val_imp = pd.DataFrame(imputer.transform(X_val_raw), columns=cols)
        X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=cols)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=cols)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val_imp), columns=cols)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=cols)

        # Save to Cache
        X_train_scaled.to_parquet(paths["train"], index=False)
        X_val_scaled.to_parquet(paths["val"], index=False)
        X_test_scaled.to_parquet(paths["test"], index=False)

        return X_train_scaled, X_val_scaled, X_test_scaled

    @time_execution
    def get_lexical_sparse_view(self, load_cached_data=True):
        """
        Generates the Lexical Sparse View (TF-IDF of Request Text).
        """
        view_name = "lexical_sparse"
        ext = ".npz"

        paths = {
            "train": self._get_cache_path(view_name, "train", ext),
            "val": self._get_cache_path(view_name, "val", ext),
            "test": self._get_cache_path(view_name, "test", ext),
        }

        if load_cached_data and all(os.path.exists(p) for p in paths.values()):
            print(f"Loading cached {view_name} view...")
            return (
                scipy.sparse.load_npz(paths["train"]),
                scipy.sparse.load_npz(paths["val"]),
                scipy.sparse.load_npz(paths["test"]),
            )

        print(f"Computing {view_name} view...")

        # Prepare Text
        train_text = self.train_df[TEXT_COL].fillna("").astype(str)
        val_text = self.val_df[TEXT_COL].fillna("").astype(str)
        test_text = self.test_df[TEXT_COL].fillna("").astype(str)

        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=TFIDF_SUBLINEAR_TF,
            stop_words="english",
        )

        X_train = vectorizer.fit_transform(train_text)
        X_val = vectorizer.transform(val_text)
        X_test = vectorizer.transform(test_text)

        # Save
        scipy.sparse.save_npz(paths["train"], X_train)
        scipy.sparse.save_npz(paths["val"], X_val)
        scipy.sparse.save_npz(paths["test"], X_test)

        return X_train, X_val, X_test

    @time_execution
    def get_lexical_dense_view(self, load_cached_data=True):
        """
        Generates the Lexical Dense View (MPNet Embeddings of Request Text).
        """
        view_name = "lexical_dense"
        ext = ".npy"

        paths = {
            "train": self._get_cache_path(view_name, "train", ext),
            "val": self._get_cache_path(view_name, "val", ext),
            "test": self._get_cache_path(view_name, "test", ext),
        }

        if load_cached_data and all(os.path.exists(p) for p in paths.values()):
            print(f"Loading cached {view_name} view...")
            return (
                np.load(paths["train"]),
                np.load(paths["val"]),
                np.load(paths["test"]),
            )

        print(f"Computing {view_name} view...")

        # Prepare Text
        train_text = self.train_df[TEXT_COL].fillna("").astype(str).tolist()
        val_text = self.val_df[TEXT_COL].fillna("").astype(str).tolist()
        test_text = self.test_df[TEXT_COL].fillna("").astype(str).tolist()

        # Load Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer: {TRANSFORMER_MODEL_NAME} on {device}")
        model = SentenceTransformer(TRANSFORMER_MODEL_NAME, device=device)

        # Encode
        X_train = model.encode(
            train_text, show_progress_bar=False, convert_to_numpy=True
        )
        X_val = model.encode(val_text, show_progress_bar=False, convert_to_numpy=True)
        X_test = model.encode(test_text, show_progress_bar=False, convert_to_numpy=True)

        # Save
        np.save(paths["train"], X_train)
        np.save(paths["val"], X_val)
        np.save(paths["test"], X_test)

        return X_train, X_val, X_test

    @time_execution
    def get_behavioral_sparse_view(self, load_cached_data=True):
        """
        Generates the Behavioral Sparse View (TF-IDF of Subreddit History).
        Treats subreddit history as a 'bag of concepts'.
        """
        view_name = "behavioral_sparse"
        ext = ".npz"

        paths = {
            "train": self._get_cache_path(view_name, "train", ext),
            "val": self._get_cache_path(view_name, "val", ext),
            "test": self._get_cache_path(view_name, "test", ext),
        }

        if load_cached_data and all(os.path.exists(p) for p in paths.values()):
            print(f"Loading cached {view_name} view...")
            return (
                scipy.sparse.load_npz(paths["train"]),
                scipy.sparse.load_npz(paths["val"]),
                scipy.sparse.load_npz(paths["test"]),
            )

        print(f"Computing {view_name} view...")

        # Serialize Subreddits: List -> Space separated string
        def serialize_subreddits(series):
            # Handle potential None or non-list types safely
            return series.apply(lambda x: " ".join(x) if isinstance(x, list) else "")

        train_subs = serialize_subreddits(self.train_df[SUBREDDIT_COL])
        val_subs = serialize_subreddits(self.val_df[SUBREDDIT_COL])
        test_subs = serialize_subreddits(self.test_df[SUBREDDIT_COL])

        # Vectorize (Treating subreddit names as tokens)
        # We use simpler settings than text: just unigrams, no stop words
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 1),
            min_df=2,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",  # Allow single char subreddits
        )

        X_train = vectorizer.fit_transform(train_subs)
        X_val = vectorizer.transform(val_subs)
        X_test = vectorizer.transform(test_subs)

        # Save
        scipy.sparse.save_npz(paths["train"], X_train)
        scipy.sparse.save_npz(paths["val"], X_val)
        scipy.sparse.save_npz(paths["test"], X_test)

        return X_train, X_val, X_test

    @time_execution
    def get_behavioral_dense_view(self, load_cached_data=True):
        """
        Generates the Behavioral Dense View (MPNet Embeddings of Subreddit History).
        Treats the list of subreddits as a semantic sentence describing the user's persona.
        """
        view_name = "behavioral_dense"
        ext = ".npy"

        paths = {
            "train": self._get_cache_path(view_name, "train", ext),
            "val": self._get_cache_path(view_name, "val", ext),
            "test": self._get_cache_path(view_name, "test", ext),
        }

        if load_cached_data and all(os.path.exists(p) for p in paths.values()):
            print(f"Loading cached {view_name} view...")
            return (
                np.load(paths["train"]),
                np.load(paths["val"]),
                np.load(paths["test"]),
            )

        print(f"Computing {view_name} view...")

        # Serialize Subreddits
        def serialize_subreddits(series):
            return series.apply(lambda x: " ".join(x) if isinstance(x, list) else "")

        train_subs = serialize_subreddits(self.train_df[SUBREDDIT_COL]).tolist()
        val_subs = serialize_subreddits(self.val_df[SUBREDDIT_COL]).tolist()
        test_subs = serialize_subreddits(self.test_df[SUBREDDIT_COL]).tolist()

        # Load Model (Reuse if already loaded in memory would be optimization, but clean init here)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"Loading SentenceTransformer for Behavioral View: {TRANSFORMER_MODEL_NAME} on {device}"
        )
        model = SentenceTransformer(TRANSFORMER_MODEL_NAME, device=device)

        # Encode
        X_train = model.encode(
            train_subs, show_progress_bar=False, convert_to_numpy=True
        )
        X_val = model.encode(val_subs, show_progress_bar=False, convert_to_numpy=True)
        X_test = model.encode(test_subs, show_progress_bar=False, convert_to_numpy=True)

        # Save
        np.save(paths["train"], X_train)
        np.save(paths["val"], X_val)
        np.save(paths["test"], X_test)

        return X_train, X_val, X_test
