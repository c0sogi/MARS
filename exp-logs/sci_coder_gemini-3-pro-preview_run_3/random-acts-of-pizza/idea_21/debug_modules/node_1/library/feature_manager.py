import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers import SentenceTransformer
import torch

from library import config, utils, data_factory


class FeatureExtractor:
    """
    Central component for generating feature views for the Hex-View Stacking Ensemble.

    Generates:
    1. Lexical View (Sparse TF-IDF of Request Text)
    2. Behavioral View (Sparse TF-IDF of Subreddit History)
    3. Semantic Text View (Dense Embeddings of Request Text)
    4. Semantic History View (Dense Embeddings of Subreddit History)
    5. Metadata View (Numerical Features + Cross-Modal Interaction)
    """

    def __init__(self):
        self.logger = utils.get_logger("FeatureExtractor")
        self.cleaner = data_factory.DataCleaner()
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file names
        self.cache_files = {
            "train": {
                "lexical": "X_train_lexical.npz",
                "behavioral": "X_train_behavioral.npz",
                "semantic_text": "X_train_semantic_text.npy",
                "semantic_history": "X_train_semantic_history.npy",
                "metadata": "X_train_metadata.npy",
                "y": "y_train.npy",
            },
            "val": {
                "lexical": "X_val_lexical.npz",
                "behavioral": "X_val_behavioral.npz",
                "semantic_text": "X_val_semantic_text.npy",
                "semantic_history": "X_val_semantic_history.npy",
                "metadata": "X_val_metadata.npy",
                "y": "y_val.npy",
            },
            "test": {
                "lexical": "X_test_lexical.npz",
                "behavioral": "X_test_behavioral.npz",
                "semantic_text": "X_test_semantic_text.npy",
                "semantic_history": "X_test_semantic_history.npy",
                "metadata": "X_test_metadata.npy",
                # No target for test
            },
        }

    def extract_features(self, load_cached_data=True, sample_size=None):
        """
        Orchestrates the feature extraction process.

        Args:
            load_cached_data (bool): If True, attempts to load processed features from disk.
            sample_size (int, optional): For debugging, limits input data size.

        Returns:
            dict: A dictionary containing 'train', 'val', 'test' sub-dictionaries with feature matrices.
        """
        # 1. Check Cache
        if load_cached_data and self._check_cache_exists():
            self.logger.info("Loading features from cache...")
            return self._load_from_cache()

        self.logger.info(
            "Cache miss or force reload. Computing features from scratch..."
        )

        # 2. Load and Clean Data
        # We load raw data first
        df_train = data_factory.DataLoader.load_train(sample_size)
        df_val = data_factory.DataLoader.load_val(sample_size)
        df_test = data_factory.DataLoader.load_test(sample_size)

        # Clean data (handles list->string conversion, basic nulls)
        df_train = self.cleaner.clean_data(df_train, "train", load_cached_data)
        df_val = self.cleaner.clean_data(df_val, "val", load_cached_data)
        df_test = self.cleaner.clean_data(df_test, "test", load_cached_data)

        # 3. Generate Dense Embeddings (Semantic Views)
        # We do this first because Metadata needs the Interaction score derived from these
        self.logger.info("Generating Dense Semantic Embeddings...")
        sem_text, sem_hist = self._generate_embeddings(df_train, df_val, df_test)

        # 4. Compute Cross-Modal Interaction
        self.logger.info("Computing Cross-Modal Interaction (Cosine Similarity)...")
        interaction_train = self._compute_interaction(
            sem_text["train"], sem_hist["train"]
        )
        interaction_val = self._compute_interaction(sem_text["val"], sem_hist["val"])
        interaction_test = self._compute_interaction(sem_text["test"], sem_hist["test"])

        # 5. Process Metadata (including Interaction)
        self.logger.info("Processing Metadata...")
        meta_feats = self._process_metadata(
            df_train,
            df_val,
            df_test,
            interaction_train,
            interaction_val,
            interaction_test,
        )

        # 6. Generate Sparse Features (TF-IDF)
        self.logger.info("Generating Sparse TF-IDF Features...")
        lexical_feats, behavioral_feats = self._generate_tfidf(
            df_train, df_val, df_test
        )

        # 7. Extract Targets
        y_train = df_train[config.TARGET_COL].values.astype(int)
        y_val = df_val[config.TARGET_COL].values.astype(int)

        # 8. Construct Output & Save to Cache
        features = {
            "train": {
                "lexical": lexical_feats["train"],
                "behavioral": behavioral_feats["train"],
                "semantic_text": sem_text["train"],
                "semantic_history": sem_hist["train"],
                "metadata": meta_feats["train"],
                "y": y_train,
            },
            "val": {
                "lexical": lexical_feats["val"],
                "behavioral": behavioral_feats["val"],
                "semantic_text": sem_text["val"],
                "semantic_history": sem_hist["val"],
                "metadata": meta_feats["val"],
                "y": y_val,
            },
            "test": {
                "lexical": lexical_feats["test"],
                "behavioral": behavioral_feats["test"],
                "semantic_text": sem_text["test"],
                "semantic_history": sem_hist["test"],
                "metadata": meta_feats["test"],
            },
        }

        self._save_to_cache(features)
        return features

    def _generate_embeddings(self, df_train, df_val, df_test):
        """
        Generates dense embeddings using SentenceTransformer.
        """
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(
            f"Loading Embedding Model {config.EMBEDDING_MODEL} on {device}..."
        )

        model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

        # Helper to encode a series
        def encode_series(series):
            # Ensure strings
            texts = series.fillna("").astype(str).tolist()
            # Encode
            return model.encode(
                texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )

        # Text Embeddings
        self.logger.info("Encoding Request Text...")
        sem_text = {
            "train": encode_series(df_train[config.TEXT_COL]),
            "val": encode_series(df_val[config.TEXT_COL]),
            "test": encode_series(df_test[config.TEXT_COL]),
        }

        # History Embeddings
        self.logger.info("Encoding Subreddit History...")
        sem_hist = {
            "train": encode_series(df_train[config.HISTORY_COL]),
            "val": encode_series(df_val[config.HISTORY_COL]),
            "test": encode_series(df_test[config.HISTORY_COL]),
        }

        return sem_text, sem_hist

    def _compute_interaction(self, emb_text, emb_hist):
        """
        Computes row-wise cosine similarity between text and history embeddings.
        Returns a column vector (N, 1).
        """
        # paired_cosine_distances returns distance (1 - similarity)
        # We want similarity, so 1 - distance
        # Result is shape (N,)
        dists = paired_cosine_distances(emb_text, emb_hist)
        sims = 1.0 - dists
        return sims.reshape(-1, 1)

    def _process_metadata(
        self, df_train, df_val, df_test, int_train, int_val, int_test
    ):
        """
        Selects numerical columns, imputes, adds interaction, and scales.
        """
        # 1. Identify Numerical Columns
        # Exclude ID, Target, Text, History, Source, and Retrieval-time leakage
        all_cols = df_train.columns
        exclude = set(config.EXCLUDE_COLS)

        # Filter for numeric types
        numeric_cols = df_train.select_dtypes(include=["number"]).columns.tolist()

        # Filter exclusions and leakage
        final_cols = []
        for c in numeric_cols:
            if c in exclude:
                continue
            if c.endswith(config.RETRIEVAL_SUFFIX):
                continue
            final_cols.append(c)

        self.logger.info(f"Selected {len(final_cols)} metadata columns: {final_cols}")

        # 2. Extract raw matrices
        X_train = df_train[final_cols].values
        X_val = df_val[final_cols].values
        X_test = df_test[final_cols].values

        # 3. Impute Missing Values (Median)
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)

        # 4. Append Interaction Feature
        # Concatenate along the last axis
        X_train = np.hstack([X_train, int_train])
        X_val = np.hstack([X_val, int_val])
        X_test = np.hstack([X_test, int_test])

        # 5. Scale Features (StandardScaler)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return {"train": X_train, "val": X_val, "test": X_test}

    def _generate_tfidf(self, df_train, df_val, df_test):
        """
        Generates sparse TF-IDF matrices for text and history.
        """
        # Lexical (Text)
        self.logger.info("Vectorizing Text (Lexical View)...")
        tfidf_text = TfidfVectorizer(**config.TFIDF_PARAMS)
        lex_train = tfidf_text.fit_transform(df_train[config.TEXT_COL].fillna(""))
        lex_val = tfidf_text.transform(df_val[config.TEXT_COL].fillna(""))
        lex_test = tfidf_text.transform(df_test[config.TEXT_COL].fillna(""))

        # Behavioral (History)
        self.logger.info("Vectorizing History (Behavioral View)...")
        # Reuse params but maybe different max_features? Using same config for simplicity/robustness
        tfidf_hist = TfidfVectorizer(**config.TFIDF_PARAMS)
        beh_train = tfidf_hist.fit_transform(df_train[config.HISTORY_COL].fillna(""))
        beh_val = tfidf_hist.transform(df_val[config.HISTORY_COL].fillna(""))
        beh_test = tfidf_hist.transform(df_test[config.HISTORY_COL].fillna(""))

        return (
            {"train": lex_train, "val": lex_val, "test": lex_test},
            {"train": beh_train, "val": beh_val, "test": beh_test},
        )

    def _check_cache_exists(self):
        """Checks if all required cache files exist."""
        for split, files in self.cache_files.items():
            for key, filename in files.items():
                path = os.path.join(self.cache_dir, filename)
                if not os.path.exists(path):
                    return False
        return True

    def _save_to_cache(self, features):
        """Saves feature matrices to disk."""
        self.logger.info(f"Saving features to {self.cache_dir}...")

        for split, data_dict in features.items():
            file_map = self.cache_files[split]
            for key, matrix in data_dict.items():
                if key not in file_map:
                    continue
                path = os.path.join(self.cache_dir, file_map[key])

                if sp.issparse(matrix):
                    sp.save_npz(path, matrix)
                else:
                    np.save(path, matrix)

    def _load_from_cache(self):
        """Loads feature matrices from disk."""
        features = {"train": {}, "val": {}, "test": {}}

        for split, files in self.cache_files.items():
            for key, filename in files.items():
                path = os.path.join(self.cache_dir, filename)
                if filename.endswith(".npz"):
                    features[split][key] = sp.load_npz(path)
                else:
                    features[split][key] = np.load(path)

        return features
