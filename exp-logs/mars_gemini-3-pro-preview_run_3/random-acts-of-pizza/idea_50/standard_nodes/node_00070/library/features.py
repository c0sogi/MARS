import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import torch

from library.config import (
    LEXICAL_TFIDF_PARAMS,
    COMMUNITY_TFIDF_PARAMS,
    ALLOW_LIST_METADATA,
    TEXT_COLS,
    SUBREDDIT_COL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    WORKING_DIR,
    SEED,
)
from library.utils import Timer, set_seed


class FeatureEngineer:
    """
    Handles feature engineering for the Hex-View Stacking Ensemble.
    Generates Lexical, Community, Semantic, and Metadata features.
    Implements caching to avoid redundant computation.
    """

    def __init__(self):
        set_seed(SEED)

        # Initialize Transformers
        # Lexical: TF-IDF on Title + Body
        self.lexical_vectorizer = TfidfVectorizer(**LEXICAL_TFIDF_PARAMS)

        # Community: TF-IDF on Subreddit History
        # Ensure lowercase is False because input is list of strings, not raw text
        community_params = COMMUNITY_TFIDF_PARAMS.copy()
        if "lowercase" not in community_params:
            community_params["lowercase"] = False
        self.community_vectorizer = TfidfVectorizer(**community_params)

        # Metadata: Imputer and Scaler
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # Semantic model is loaded on demand to save memory/startup time if cached
        self.embedding_model = None

    def _get_text_data(self, df: pd.DataFrame) -> pd.Series:
        """Concatenates title and edit-aware body text."""
        # Ensure columns exist and are strings (handled by data_loader, but safety first)
        title = df["request_title"].fillna("").astype(str)
        body = df["request_text_edit_aware"].fillna("").astype(str)
        return title + " " + body

    def _get_community_data(self, df: pd.DataFrame) -> pd.Series:
        """Returns the subreddit list column."""
        # data_loader ensures this is a list, but we ensure it's iterable for sklearn
        return df[SUBREDDIT_COL]

    def _get_metadata_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts allow-listed metadata columns."""
        # Filter for available columns
        available_cols = [c for c in ALLOW_LIST_METADATA if c in df.columns]
        return df[available_cols]

    def _compute_lexical(self, train_df, val_df, test_df):
        with Timer("Computing Lexical Features"):
            train_text = self._get_text_data(train_df)
            val_text = self._get_text_data(val_df)
            test_text = self._get_text_data(test_df)

            # Fit on train, transform all
            X_train = self.lexical_vectorizer.fit_transform(train_text)
            X_val = self.lexical_vectorizer.transform(val_text)
            X_test = self.lexical_vectorizer.transform(test_text)

            return X_train, X_val, X_test

    def _compute_community(self, train_df, val_df, test_df):
        with Timer("Computing Community Features"):
            train_comm = self._get_community_data(train_df)
            val_comm = self._get_community_data(val_df)
            test_comm = self._get_community_data(test_df)

            # Fit on train, transform all
            X_train = self.community_vectorizer.fit_transform(train_comm)
            X_val = self.community_vectorizer.transform(val_comm)
            X_test = self.community_vectorizer.transform(test_comm)

            return X_train, X_val, X_test

    def _compute_semantic(self, train_df, val_df, test_df):
        with Timer("Computing Semantic Features"):
            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME, device=device
            )

            train_text = self._get_text_data(train_df).tolist()
            val_text = self._get_text_data(val_df).tolist()
            test_text = self._get_text_data(test_df).tolist()

            # Encode
            X_train = self.embedding_model.encode(
                train_text,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            X_val = self.embedding_model.encode(
                val_text,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            X_test = self.embedding_model.encode(
                test_text,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return X_train, X_val, X_test

    def _compute_metadata(self, train_df, val_df, test_df):
        with Timer("Computing Metadata Features"):
            X_train_raw = self._get_metadata_data(train_df)
            X_val_raw = self._get_metadata_data(val_df)
            X_test_raw = self._get_metadata_data(test_df)

            # Impute
            X_train_imp = self.imputer.fit_transform(X_train_raw)
            X_val_imp = self.imputer.transform(X_val_raw)
            X_test_imp = self.imputer.transform(X_test_raw)

            # Scale
            X_train = self.scaler.fit_transform(X_train_imp)
            X_val = self.scaler.transform(X_val_imp)
            X_test = self.scaler.transform(X_test_imp)

            return X_train, X_val, X_test

    def _get_cache_paths(self):
        """Returns a dictionary of expected cache file paths."""
        partitions = ["train", "val", "test"]
        types = {
            "lexical": "npz",
            "community": "npz",
            "semantic": "npy",
            "metadata": "npy",
        }
        paths = {}
        for p in partitions:
            paths[p] = {}
            for t, ext in types.items():
                filename = f"X_{p}_{t}.{ext}"
                paths[p][t] = os.path.join(WORKING_DIR, filename)
        return paths

    def _check_cache(self, paths):
        """Checks if all cache files exist."""
        for p in paths:
            for t in paths[p]:
                if not os.path.exists(paths[p][t]):
                    return False
        return True

    def _save_to_cache(self, data_dict, paths):
        """Saves computed features to cache."""
        with Timer("Saving Features to Cache"):
            os.makedirs(WORKING_DIR, exist_ok=True)
            for p in data_dict:
                for t in data_dict[p]:
                    data = data_dict[p][t]
                    path = paths[p][t]
                    if path.endswith(".npz"):
                        scipy.sparse.save_npz(path, data)
                    else:
                        np.save(path, data)

    def _load_from_cache(self, paths):
        """Loads features from cache."""
        with Timer("Loading Features from Cache"):
            data_dict = {"train": {}, "val": {}, "test": {}}
            for p in paths:
                for t in paths[p]:
                    path = paths[p][t]
                    if path.endswith(".npz"):
                        data_dict[p][t] = scipy.sparse.load_npz(path)
                    else:
                        data_dict[p][t] = np.load(path)
            return data_dict["train"], data_dict["val"], data_dict["test"]

    def generate_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main method to generate all features.

        Args:
            train_df, val_df, test_df: DataFrames for each split.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (train_feats, val_feats, test_feats)
                   Each element is a dict with keys: 'lexical', 'community', 'semantic', 'metadata'.
        """
        paths = self._get_cache_paths()

        if load_cached_data and self._check_cache(paths):
            try:
                train_feats, val_feats, test_feats = self._load_from_cache(paths)
                # Validate dimensions against current data inputs (Cite debug_lesson_2)
                if (
                    train_feats["metadata"].shape[0] == len(train_df)
                    and val_feats["metadata"].shape[0] == len(val_df)
                    and test_feats["metadata"].shape[0] == len(test_df)
                ):
                    print("Features loaded from cache and validated.")
                    return train_feats, val_feats, test_feats
                else:
                    print("Cached features dimension mismatch. Recomputing...")
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        # Compute features
        train_lex, val_lex, test_lex = self._compute_lexical(train_df, val_df, test_df)
        train_comm, val_comm, test_comm = self._compute_community(
            train_df, val_df, test_df
        )
        train_sem, val_sem, test_sem = self._compute_semantic(train_df, val_df, test_df)
        train_meta, val_meta, test_meta = self._compute_metadata(
            train_df, val_df, test_df
        )

        # Organize into dictionaries
        train_feats = {
            "lexical": train_lex,
            "community": train_comm,
            "semantic": train_sem,
            "metadata": train_meta,
        }
        val_feats = {
            "lexical": val_lex,
            "community": val_comm,
            "semantic": val_sem,
            "metadata": val_meta,
        }
        test_feats = {
            "lexical": test_lex,
            "community": test_comm,
            "semantic": test_sem,
            "metadata": test_meta,
        }

        full_data = {"train": train_feats, "val": val_feats, "test": test_feats}

        # Save to cache
        self._save_to_cache(full_data, paths)

        return train_feats, val_feats, test_feats
