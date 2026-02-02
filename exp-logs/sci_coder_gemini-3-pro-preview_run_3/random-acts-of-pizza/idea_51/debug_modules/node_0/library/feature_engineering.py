import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import torch

from library import config
from library import utils
from library import data_loader


class FeaturePipeline:
    """
    Manages the creation of feature sets for the Hept-View Ensemble.
    Generates Lexical, Behavioral, Semantic, and Contextual feature views.
    Implements strict caching to disk.
    """

    def __init__(self, load_cached_data: bool = True):
        self.load_cached_data = load_cached_data
        self.cache_dir = config.WORKING_DIR

        # Map logical names to filenames
        self.cache_files = {
            "X_train_lexical": "X_train_lexical.npz",
            "X_test_lexical": "X_test_lexical.npz",
            "X_train_behavioral": "X_train_behavioral.npz",
            "X_test_behavioral": "X_test_behavioral.npz",
            "X_train_semantic": "X_train_semantic.npy",
            "X_test_semantic": "X_test_semantic.npy",
            "X_train_contextual": "X_train_contextual.npy",
            "X_test_contextual": "X_test_contextual.npy",
            "y_train": "y_train.npy",
            "train_ids": "train_ids.npy",
            "test_ids": "test_ids.npy",
        }

    def _check_cache_exists(self) -> bool:
        """Checks if all required cache files exist."""
        for filename in self.cache_files.values():
            if not os.path.exists(os.path.join(self.cache_dir, filename)):
                return False
        return True

    def _load_from_cache(self) -> dict:
        """Loads all feature matrices from disk."""
        print("[FeaturePipeline] Loading features from cache...")
        data = {}
        for key, filename in self.cache_files.items():
            path = os.path.join(self.cache_dir, filename)
            if filename.endswith(".npz"):
                data[key] = sp.load_npz(path)
            else:
                data[key] = np.load(path, allow_pickle=True)
        return data

    def _save_to_cache(self, data: dict) -> None:
        """Saves all feature matrices to disk."""
        print("[FeaturePipeline] Saving features to cache...")
        os.makedirs(self.cache_dir, exist_ok=True)
        for key, filename in self.cache_files.items():
            path = os.path.join(self.cache_dir, filename)
            if key not in data:
                continue
            if filename.endswith(".npz"):
                sp.save_npz(path, data[key])
            else:
                np.save(path, data[key])

    def _get_text_data(self, df: pd.DataFrame) -> list:
        """Concatenates title and edit-aware text."""
        # Fill NAs with empty string to avoid errors
        title = df["request_title"].fillna("").astype(str)
        text = df["request_text_edit_aware"].fillna("").astype(str)
        return (title + " " + text).tolist()

    def _get_lexical_features(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """Generates Sparse TF-IDF features from text content."""
        print("[FeaturePipeline] Generating Lexical features...")
        train_text = self._get_text_data(df_train)
        test_text = self._get_text_data(df_test)

        vectorizer = TfidfVectorizer(**config.LEXICAL_VECTORIZER_PARAMS)
        X_train = vectorizer.fit_transform(train_text)
        X_test = vectorizer.transform(test_text)

        return X_train, X_test

    def _get_behavioral_features(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """Generates Sparse TF-IDF features from subreddit history (Bag-of-Concepts)."""
        print("[FeaturePipeline] Generating Behavioral features...")

        def process_subreddits(series):
            # Join list of strings into space-separated string
            # Handle cases where it might not be a list (though schema says array)
            return series.apply(lambda x: " ".join(x) if isinstance(x, list) else "")

        train_subs = process_subreddits(df_train["requester_subreddits_at_request"])
        test_subs = process_subreddits(df_test["requester_subreddits_at_request"])

        vectorizer = TfidfVectorizer(**config.COMMUNITY_VECTORIZER_PARAMS)
        X_train = vectorizer.fit_transform(train_subs)
        X_test = vectorizer.transform(test_subs)

        return X_train, X_test

    def _get_semantic_features(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """Generates Dense Embedding features using Sentence Transformers."""
        print(
            f"[FeaturePipeline] Generating Semantic features ({config.EMBEDDING_MODEL})..."
        )

        train_text = self._get_text_data(df_train)
        test_text = self._get_text_data(df_test)

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[FeaturePipeline] Embedding inference on {device}")

        model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

        # Encode
        X_train = model.encode(
            train_text, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        X_test = model.encode(
            test_text, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        return X_train, X_test

    def _get_contextual_features(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """Generates Dense Metadata features (Imputed and Scaled)."""
        print("[FeaturePipeline] Generating Contextual features...")

        features = config.METADATA_FEATURES

        # Extract raw matrices
        X_train_raw = df_train[features].values
        X_test_raw = df_test[features].values

        # Impute
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        return X_train_scaled, X_test_scaled

    def run(self) -> dict:
        """
        Executes the feature generation pipeline.
        Returns a dictionary containing all feature matrices and targets.
        """
        utils.set_seed()

        # 1. Check Cache
        if self.load_cached_data and self._check_cache_exists():
            return self._load_from_cache()

        # 2. Load Data
        df_train, df_test = data_loader.load_datasets(
            load_cached_data=self.load_cached_data
        )

        # 3. Generate Features
        with utils.Timer("Feature Generation"):
            # Lexical (Sparse)
            X_train_lex, X_test_lex = self._get_lexical_features(df_train, df_test)

            # Behavioral (Sparse)
            X_train_beh, X_test_beh = self._get_behavioral_features(df_train, df_test)

            # Semantic (Dense)
            X_train_sem, X_test_sem = self._get_semantic_features(df_train, df_test)

            # Contextual (Dense Metadata)
            X_train_ctx, X_test_ctx = self._get_contextual_features(df_train, df_test)

            # Targets and IDs
            y_train = df_train[config.TARGET_COL].values.astype(int)
            train_ids = df_train[config.ID_COL].values
            test_ids = df_test[config.ID_COL].values

        # 4. Construct Data Dictionary
        data = {
            "X_train_lexical": X_train_lex,
            "X_test_lexical": X_test_lex,
            "X_train_behavioral": X_train_beh,
            "X_test_behavioral": X_test_beh,
            "X_train_semantic": X_train_sem,
            "X_test_semantic": X_test_sem,
            "X_train_contextual": X_train_ctx,
            "X_test_contextual": X_test_ctx,
            "y_train": y_train,
            "train_ids": train_ids,
            "test_ids": test_ids,
        }

        # 5. Save to Cache
        self._save_to_cache(data)

        return data
