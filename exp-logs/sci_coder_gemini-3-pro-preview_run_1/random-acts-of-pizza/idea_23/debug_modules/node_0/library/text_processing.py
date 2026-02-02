import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
from library import config


class TfidfPipeline:
    """
    Manages TF-IDF vectorization for the Random Forest stream.
    Combines request title and body, generates high-fidelity features, and caches results.
    """

    def __init__(self, max_features=None, ngram_range=(1, 2)):
        self.max_features = max_features if max_features else config.TFIDF_MAX_FEATURES
        self.ngram_range = ngram_range
        self.cache_path = os.path.join(config.WORKING_DIR, "tfidf_features.npz")

    def _prepare_text(self, df):
        """
        Combines title and body text into a single string for vectorization.
        Handles missing values and performs basic normalization.
        """
        title = df[config.TEXT_COL_TITLE].fillna("").astype(str)
        body = df[config.TEXT_COL_BODY].fillna("").astype(str)
        return (title + " " + body).str.lower().str.strip()

    def run(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Generates or loads TF-IDF features for all data splits.

        Args:
            df_train, df_val, df_test: DataFrames containing text columns.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Dictionary with 'train', 'val', 'test' keys containing dense numpy arrays.
        """
        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading TF-IDF features from {self.cache_path}...")
            try:
                data = np.load(self.cache_path)
                return {
                    "train": data["train"],
                    "val": data["val"],
                    "test": data["test"],
                }
            except Exception as e:
                print(f"Failed to load TF-IDF cache: {e}. Regenerating...")

        # 2. Generate Features
        print("Generating TF-IDF features...")
        text_train = self._prepare_text(df_train)
        text_val = self._prepare_text(df_val)
        text_test = self._prepare_text(df_test)

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
        )

        # Fit on training data only to prevent leakage
        X_train = vectorizer.fit_transform(text_train).toarray()
        X_val = vectorizer.transform(text_val).toarray()
        X_test = vectorizer.transform(text_test).toarray()

        # 3. Save to Cache
        print(f"Saving TF-IDF features to {self.cache_path}...")
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        np.savez(self.cache_path, train=X_train, val=X_val, test=X_test)

        return {"train": X_train, "val": X_val, "test": X_test}


class SbertEncoder:
    """
    Manages SBERT embeddings for the MLP stream.
    Generates embeddings for the Request text and the User's Subreddit History.
    """

    def __init__(self, model_name=None, batch_size=32):
        self.model_name = model_name if model_name else config.SBERT_MODEL_NAME
        self.batch_size = batch_size
        self.device = config.DEVICE
        self.history_max_len = 50  # Fixed sequence length for history

    def _load_model(self):
        print(f"Loading SBERT model: {self.model_name}...")
        return SentenceTransformer(self.model_name, device=self.device)

    def _prepare_request_text(self, df):
        title = df[config.TEXT_COL_TITLE].fillna("").astype(str)
        body = df[config.TEXT_COL_BODY].fillna("").astype(str)
        return (title + " " + body).tolist()

    def generate_request_embeddings(
        self, df_train, df_val, df_test, load_cached_data=True
    ):
        """
        Generates embeddings for the request text (title + body).
        """
        path = config.CACHE_SBERT_REQUEST

        if load_cached_data and os.path.exists(path):
            print(f"Loading SBERT request embeddings from {path}...")
            try:
                data = np.load(path)
                return {
                    "train": data["train"],
                    "val": data["val"],
                    "test": data["test"],
                }
            except Exception as e:
                print(f"Failed to load SBERT request cache: {e}. Regenerating...")

        print("Generating SBERT request embeddings...")
        model = self._load_model()

        # Encode
        emb_train = model.encode(
            self._prepare_request_text(df_train),
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        emb_val = model.encode(
            self._prepare_request_text(df_val),
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        emb_test = model.encode(
            self._prepare_request_text(df_test),
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Save
        print(f"Saving SBERT request embeddings to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, train=emb_train, val=emb_val, test=emb_test)

        return {"train": emb_train, "val": emb_val, "test": emb_test}

    def generate_history_embeddings(
        self, df_train, df_val, df_test, load_cached_data=True
    ):
        """
        Generates sequence embeddings for user history.
        Encodes unique subreddits and maps them to user sequences, padding to fixed length.
        """
        path = config.CACHE_SBERT_HISTORY

        if load_cached_data and os.path.exists(path):
            print(f"Loading SBERT history embeddings from {path}...")
            try:
                data = np.load(path)
                return {
                    "train": data["train"],
                    "val": data["val"],
                    "test": data["test"],
                }
            except Exception as e:
                print(f"Failed to load SBERT history cache: {e}. Regenerating...")

        print("Generating SBERT history embeddings...")
        model = self._load_model()

        # 1. Collect all unique subreddits to encode efficiently
        all_dfs = [df_train, df_val, df_test]
        unique_subs = set()
        for df in all_dfs:
            if config.HISTORY_COL in df.columns:
                for sub_list in df[config.HISTORY_COL]:
                    if isinstance(sub_list, list):
                        unique_subs.update(sub_list)

        unique_subs = list(unique_subs)
        print(f"Found {len(unique_subs)} unique subreddits in history.")

        # 2. Encode unique subreddits
        if not unique_subs:
            sub_emb_map = {}
            embedding_dim = 384  # Default for MiniLM
        else:
            sub_embeddings = model.encode(
                unique_subs, batch_size=self.batch_size, show_progress_bar=False
            )
            sub_emb_map = {sub: emb for sub, emb in zip(unique_subs, sub_embeddings)}
            embedding_dim = sub_embeddings.shape[1]

        # 3. Map back to users and pad sequences
        def process_history(df):
            if config.HISTORY_COL not in df.columns:
                return np.zeros(
                    (len(df), self.history_max_len, embedding_dim), dtype=np.float32
                )

            batch_seqs = []
            for sub_list in df[config.HISTORY_COL]:
                if not isinstance(sub_list, list):
                    sub_list = []

                # Truncate to max len
                sub_list = sub_list[: self.history_max_len]

                # Get embeddings for subreddits in history
                seq = [sub_emb_map[s] for s in sub_list if s in sub_emb_map]

                # Pad with zero vectors if sequence is shorter than max len
                if len(seq) < self.history_max_len:
                    padding_needed = self.history_max_len - len(seq)
                    padding = [
                        np.zeros(embedding_dim, dtype=np.float32)
                    ] * padding_needed
                    seq.extend(padding)

                batch_seqs.append(np.array(seq))

            return np.array(batch_seqs, dtype=np.float32)

        hist_train = process_history(df_train)
        hist_val = process_history(df_val)
        hist_test = process_history(df_test)

        # Save
        print(f"Saving SBERT history embeddings to {path}...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, train=hist_train, val=hist_val, test=hist_test)

        return {"train": hist_train, "val": hist_val, "test": hist_test}
