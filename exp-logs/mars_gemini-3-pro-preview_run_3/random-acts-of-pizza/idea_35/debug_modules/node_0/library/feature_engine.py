import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import Timer, set_seed


class FeaturePipeline:
    """
    Orchestrates feature engineering for the Unified Interaction-Aware Stacking Ensemble.
    Generates Sparse (TF-IDF), Dense (Embeddings), and Metadata views with caching.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)
        set_seed(Config.RANDOM_STATE)

    def _get_cache_paths(self, prefix, extension):
        """Helper to generate cache file paths."""
        return {
            "train": os.path.join(self.working_dir, f"X_train_{prefix}.{extension}"),
            "val": os.path.join(self.working_dir, f"X_val_{prefix}.{extension}"),
            "test": os.path.join(self.working_dir, f"X_test_{prefix}.{extension}"),
        }

    def _check_cache(self, paths):
        """Checks if all files in the paths dictionary exist."""
        return all(os.path.exists(p) for p in paths.values())

    def create_sparse_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates TF-IDF matrices for Unified, Lexical, and Community views.
        """
        # Define views and their configurations
        views = {
            "unified": {
                "col": "text_unified",
                "vocab": Config.VOCAB_UNIFIED,
            },
            "lexical": {
                "col": "text_lexical",
                "vocab": Config.VOCAB_LEXICAL,
            },
            "community": {
                "col": "text_community",
                "vocab": Config.VOCAB_COMMUNITY,
            },
        }

        results = {}

        for view_name, config in views.items():
            paths = self._get_cache_paths(view_name, "npz")

            # 1. Try Loading
            if load_cached_data and self._check_cache(paths):
                print(f"Loading {view_name} sparse features from cache...")
                results[f"X_train_{view_name}"] = sparse.load_npz(paths["train"])
                results[f"X_val_{view_name}"] = sparse.load_npz(paths["val"])
                results[f"X_test_{view_name}"] = sparse.load_npz(paths["test"])
                continue

            # 2. Compute
            with Timer(f"Computing Sparse Features: {view_name}"):
                print(f"Generating {view_name} features (Vocab: {config['vocab']})...")

                # Merge base params with max_features
                params = Config.TFIDF_PARAMS.copy()
                params["max_features"] = config["vocab"]

                vectorizer = TfidfVectorizer(**params)

                # Fit on Train, Transform all
                # Fillna is handled in data_manager, but safety check here
                train_text = train_df[config["col"]].fillna("").astype(str)
                val_text = val_df[config["col"]].fillna("").astype(str)
                test_text = test_df[config["col"]].fillna("").astype(str)

                X_train = vectorizer.fit_transform(train_text)
                X_val = vectorizer.transform(val_text)
                X_test = vectorizer.transform(test_text)

                # Store in results
                results[f"X_train_{view_name}"] = X_train
                results[f"X_val_{view_name}"] = X_val
                results[f"X_test_{view_name}"] = X_test

                # Save to cache
                print(f"Saving {view_name} features to {self.working_dir}...")
                sparse.save_npz(paths["train"], X_train)
                sparse.save_npz(paths["val"], X_val)
                sparse.save_npz(paths["test"], X_test)

        return results

    def create_semantic_features(
        self, train_df, val_df, test_df, load_cached_data=True
    ):
        """
        Generates Dense Embeddings using SentenceTransformer.
        """
        paths = self._get_cache_paths("semantic", "npy")

        # 1. Try Loading
        if load_cached_data and self._check_cache(paths):
            print("Loading semantic features from cache...")
            return {
                "X_train_semantic": np.load(paths["train"]),
                "X_val_semantic": np.load(paths["val"]),
                "X_test_semantic": np.load(paths["test"]),
            }

        # 2. Compute
        with Timer("Computing Semantic Features (Embeddings)"):
            print(f"Loading Embedding Model: {Config.EMBEDDING_MODEL}")
            # We use the CPU/GPU automatically detected by torch
            model = SentenceTransformer(Config.EMBEDDING_MODEL)

            # Use Lexical view (Title + Body) for semantic embedding
            col = "text_lexical"
            train_text = train_df[col].fillna("").astype(str).tolist()
            val_text = val_df[col].fillna("").astype(str).tolist()
            test_text = test_df[col].fillna("").astype(str).tolist()

            # Encode
            # show_progress_bar=False to reduce clutter as per instructions
            X_train = model.encode(
                train_text,
                batch_size=Config.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
            )
            X_val = model.encode(
                val_text,
                batch_size=Config.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
            )
            X_test = model.encode(
                test_text,
                batch_size=Config.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
            )

            # Save
            print(f"Saving semantic features to {self.working_dir}...")
            np.save(paths["train"], X_train)
            np.save(paths["val"], X_val)
            np.save(paths["test"], X_test)

        return {
            "X_train_semantic": X_train,
            "X_val_semantic": X_val,
            "X_test_semantic": X_test,
        }

    def create_metadata_features(
        self, train_df, val_df, test_df, load_cached_data=True
    ):
        """
        Scales numerical metadata using StandardScaler.
        """
        paths = self._get_cache_paths("metadata", "npy")

        # 1. Try Loading
        if load_cached_data and self._check_cache(paths):
            print("Loading metadata features from cache...")
            return {
                "X_train_metadata": np.load(paths["train"]),
                "X_val_metadata": np.load(paths["val"]),
                "X_test_metadata": np.load(paths["test"]),
            }

        # 2. Compute
        with Timer("Computing Metadata Features"):
            # Columns are already selected and imputed in data_manager
            # We just need to extract the specific columns defined in Config
            # (In case the df has extra columns)
            meta_cols = [c for c in Config.METADATA_COLS if c in train_df.columns]

            X_train_raw = train_df[meta_cols].values.astype(np.float32)
            X_val_raw = val_df[meta_cols].values.astype(np.float32)
            X_test_raw = test_df[meta_cols].values.astype(np.float32)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_val = scaler.transform(X_val_raw)
            X_test = scaler.transform(X_test_raw)

            # Save
            print(f"Saving metadata features to {self.working_dir}...")
            np.save(paths["train"], X_train)
            np.save(paths["val"], X_val)
            np.save(paths["test"], X_test)

        return {
            "X_train_metadata": X_train,
            "X_val_metadata": X_val,
            "X_test_metadata": X_test,
        }

    def get_targets(self, train_df, val_df):
        """Extracts target variables."""
        y_train = train_df[Config.TARGET_COL].values.astype(int)
        y_val = val_df[Config.TARGET_COL].values.astype(int)
        return y_train, y_val

    def process_all(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Orchestrates the full feature generation pipeline.

        Returns:
            dict: Dictionary containing all feature matrices (X_train_*, X_val_*, X_test_*)
                  and targets (y_train, y_val).
        """
        features = {}

        # 1. Sparse Features (Unified, Lexical, Community)
        sparse_feats = self.create_sparse_features(
            train_df, val_df, test_df, load_cached_data
        )
        features.update(sparse_feats)

        # 2. Semantic Features (Embeddings)
        semantic_feats = self.create_semantic_features(
            train_df, val_df, test_df, load_cached_data
        )
        features.update(semantic_feats)

        # 3. Metadata Features
        meta_feats = self.create_metadata_features(
            train_df, val_df, test_df, load_cached_data
        )
        features.update(meta_feats)

        # 4. Targets
        y_train, y_val = self.get_targets(train_df, val_df)
        features["y_train"] = y_train
        features["y_val"] = y_val

        # 5. IDs (for submission)
        features["test_ids"] = test_df[Config.ID_COL].values

        return features
