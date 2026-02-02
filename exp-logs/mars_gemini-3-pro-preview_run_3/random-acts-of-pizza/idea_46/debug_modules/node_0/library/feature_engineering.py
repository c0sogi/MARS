import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch

from library.config import Config
from library.utils import log_info, timer, save_cache, load_cache, set_seed


class FeatureEngineer:
    """
    Generates feature views for the Hex-View Stacking Ensemble.
    Manages Lexical, Behavioral, Semantic, and Metadata feature generation
    with strict caching and reproducibility.
    """

    def __init__(self):
        set_seed(Config.RANDOM_SEED)

    def _get_cache_filenames(self, view_name):
        """Helper to generate cache filenames for train, val, and test."""
        return {
            "train": f"{view_name}_train.npy",
            "val": f"{view_name}_val.npy",
            "test": f"{view_name}_test.npy",
        }

    def _check_cache(self, filenames):
        """Checks if all required cache files exist."""
        for fname in filenames.values():
            if load_cache(fname, use_parquet=False) is None:
                return False
        return True

    def _load_from_cache_dict(self, filenames):
        """Loads all files in the filenames dictionary."""
        data = {}
        for split, fname in filenames.items():
            data[split] = load_cache(fname, use_parquet=False)
        return data["train"], data["val"], data["test"]

    def _save_to_cache_dict(self, data_map, filenames):
        """Saves data to cache using the provided filenames."""
        for split, data in data_map.items():
            save_cache(data, filenames[split], use_parquet=False)

    def get_lexical_view(self, train_df, val_df, test_df, load_from_cache=True):
        """
        Generates the Lexical View: TF-IDF vectors of concatenated Title + Body.

        Args:
            train_df, val_df, test_df: Processed DataFrames.
            load_from_cache (bool): Whether to attempt loading from cache.

        Returns:
            Tuple of numpy arrays: (X_train, X_val, X_test)
        """
        view_name = "lexical_view"
        filenames = self._get_cache_filenames(view_name)

        if load_from_cache and self._check_cache(filenames):
            log_info(f"Loading {view_name} from cache...")
            return self._load_from_cache_dict(filenames)

        with timer("Generating Lexical View (TF-IDF)"):
            log_info("Fitting TfidfVectorizer on text...")
            vectorizer = TfidfVectorizer(**Config.TFIDF_TEXT_PARAMS)

            # Fit on training data only to prevent leakage
            vectorizer.fit(train_df["text_full"])

            # Transform all splits
            # Convert to dense to satisfy allow_pickle=False in utils.load_cache
            # Given max_features=10000 and dataset size ~4k, this fits in memory (~160MB)
            X_train = (
                vectorizer.transform(train_df["text_full"]).toarray().astype(np.float32)
            )
            X_val = (
                vectorizer.transform(val_df["text_full"]).toarray().astype(np.float32)
            )
            X_test = (
                vectorizer.transform(test_df["text_full"]).toarray().astype(np.float32)
            )

            # Save to cache
            data_map = {"train": X_train, "val": X_val, "test": X_test}
            self._save_to_cache_dict(data_map, filenames)

        return X_train, X_val, X_test

    def get_behavioral_view(self, train_df, val_df, test_df, load_from_cache=True):
        """
        Generates the Behavioral View: TF-IDF vectors of Subreddit History.

        Args:
            train_df, val_df, test_df: Processed DataFrames.
            load_from_cache (bool): Whether to attempt loading from cache.

        Returns:
            Tuple of numpy arrays: (X_train, X_val, X_test)
        """
        view_name = "behavioral_view"
        filenames = self._get_cache_filenames(view_name)

        if load_from_cache and self._check_cache(filenames):
            log_info(f"Loading {view_name} from cache...")
            return self._load_from_cache_dict(filenames)

        with timer("Generating Behavioral View (Subreddit TF-IDF)"):
            log_info("Fitting TfidfVectorizer on subreddit history...")
            vectorizer = TfidfVectorizer(**Config.TFIDF_SUBREDDIT_PARAMS)

            # Fit on training data only
            vectorizer.fit(train_df["subreddit_text"])

            # Transform all splits and densify
            X_train = (
                vectorizer.transform(train_df["subreddit_text"])
                .toarray()
                .astype(np.float32)
            )
            X_val = (
                vectorizer.transform(val_df["subreddit_text"])
                .toarray()
                .astype(np.float32)
            )
            X_test = (
                vectorizer.transform(test_df["subreddit_text"])
                .toarray()
                .astype(np.float32)
            )

            # Save to cache
            data_map = {"train": X_train, "val": X_val, "test": X_test}
            self._save_to_cache_dict(data_map, filenames)

        return X_train, X_val, X_test

    def get_semantic_view(self, train_df, val_df, test_df, load_from_cache=True):
        """
        Generates the Semantic View: Dense embeddings using a pre-trained Transformer.

        Args:
            train_df, val_df, test_df: Processed DataFrames.
            load_from_cache (bool): Whether to attempt loading from cache.

        Returns:
            Tuple of numpy arrays: (X_train, X_val, X_test)
        """
        view_name = "semantic_view"
        filenames = self._get_cache_filenames(view_name)

        if load_from_cache and self._check_cache(filenames):
            log_info(f"Loading {view_name} from cache...")
            return self._load_from_cache_dict(filenames)

        with timer(f"Generating Semantic View ({Config.EMBEDDING_MODEL_NAME})"):
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log_info(f"Loading SentenceTransformer on {device}...")

            model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME, device=device)

            def encode_texts(texts):
                # Encode in batches
                return model.encode(
                    texts.tolist(),
                    batch_size=Config.EMBEDDING_BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)

            log_info("Encoding training data...")
            X_train = encode_texts(train_df["text_full"])

            log_info("Encoding validation data...")
            X_val = encode_texts(val_df["text_full"])

            log_info("Encoding test data...")
            X_test = encode_texts(test_df["text_full"])

            # Save to cache
            data_map = {"train": X_train, "val": X_val, "test": X_test}
            self._save_to_cache_dict(data_map, filenames)

        return X_train, X_val, X_test

    def get_metadata_view(self, train_df, val_df, test_df, load_from_cache=True):
        """
        Generates the Metadata View: Scaled dense features from allow-list.

        Args:
            train_df, val_df, test_df: Processed DataFrames.
            load_from_cache (bool): Whether to attempt loading from cache.

        Returns:
            Tuple of numpy arrays: (X_train, X_val, X_test)
        """
        view_name = "metadata_view"
        filenames = self._get_cache_filenames(view_name)

        if load_from_cache and self._check_cache(filenames):
            log_info(f"Loading {view_name} from cache...")
            return self._load_from_cache_dict(filenames)

        with timer("Generating Metadata View (Scaled Dense Features)"):
            feature_cols = Config.METADATA_DENSE_FEATURES

            # Extract raw numpy arrays
            # Ensure type is float32 for consistency
            X_train_raw = train_df[feature_cols].values.astype(np.float32)
            X_val_raw = val_df[feature_cols].values.astype(np.float32)
            X_test_raw = test_df[feature_cols].values.astype(np.float32)

            # Scale features
            # Fit scaler only on training data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            X_test = scaler.transform(X_test_raw)

            # Save to cache
            data_map = {"train": X_train, "val": X_val, "test": X_test}
            self._save_to_cache_dict(data_map, filenames)

        return X_train, X_val, X_test
