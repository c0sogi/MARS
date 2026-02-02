import os
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from library.config import Config


class SBERTHandler:
    """
    Handles generation of dense semantic embeddings using Sentence-BERT.
    Processes:
    1. Request Title
    2. Request Body (Edit Aware)
    3. User History (Subreddits) -> Sequence of embeddings + Centroid
    """

    def __init__(self):
        self.model_name = Config.SBERT_MODEL_NAME
        self.max_seq_len = Config.TOP_K_SUBREDDITS
        self.emb_dim = Config.MLP_EMBEDDING_DIM
        self.cache_dir = Config.CACHE_DIR
        self.device = Config.DEVICE

        # Cache file paths
        self.cache_files = {
            "train": os.path.join(self.cache_dir, "sbert_features_train.npz"),
            "val": os.path.join(self.cache_dir, "sbert_features_val.npz"),
            "test": os.path.join(self.cache_dir, "sbert_features_test.npz"),
        }

    def process_data(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates or loads SBERT embeddings for all splits.
        Returns a dictionary containing features for 'train', 'val', and 'test'.
        """
        Config.ensure_dirs()

        # Check if all cache files exist
        all_cached = all(os.path.exists(f) for f in self.cache_files.values())

        if load_cached_data and all_cached:
            print("Loading SBERT features from cache...")
            results = {}
            for split, path in self.cache_files.items():
                data = np.load(path)
                results[split] = {
                    "title": data["title"],
                    "body": data["body"],
                    "history": data["history"],
                    "history_mask": data["history_mask"],
                    "centroid": data["centroid"],
                }
            return results

        print("Generating SBERT features (Cache miss or force reload)...")

        # Initialize model
        print(f"Loading SentenceTransformer: {self.model_name}")
        model = SentenceTransformer(self.model_name, device=self.device)

        # Prepare results container
        results = {}
        splits = {"train": train_df, "val": val_df, "test": test_df}

        # 1. Optimize Subreddit Encoding
        # Gather all unique subreddits across all splits to encode in one batch
        print("Encoding unique subreddits...")
        all_subreddits = set()
        for df in splits.values():
            for sub_list in df["requester_subreddits_at_request"]:
                # sub_list is already a list due to DataLoader
                if isinstance(sub_list, list):
                    all_subreddits.update(sub_list)

        unique_subs_list = sorted(list(all_subreddits))
        if not unique_subs_list:
            # Handle edge case where no subreddits exist
            sub_to_emb = {}
            dummy_emb = np.zeros(self.emb_dim, dtype=np.float32)
        else:
            sub_embeddings = model.encode(
                unique_subs_list,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            sub_to_emb = {
                sub: emb for sub, emb in zip(unique_subs_list, sub_embeddings)
            }
            dummy_emb = np.zeros(self.emb_dim, dtype=np.float32)

        # 2. Process each split
        for split_name, df in splits.items():
            print(f"Processing {split_name} split...")

            # A. Title Embeddings
            titles = df["request_title"].fillna("").astype(str).tolist()
            title_embs = model.encode(
                titles, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )

            # B. Body Embeddings
            bodies = df["request_text_edit_aware"].fillna("").astype(str).tolist()
            body_embs = model.encode(
                bodies, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )

            # C. History Embeddings (Sequence + Centroid)
            n_samples = len(df)
            history_seq = np.zeros(
                (n_samples, self.max_seq_len, self.emb_dim), dtype=np.float32
            )
            history_mask = np.zeros((n_samples, self.max_seq_len), dtype=np.float32)
            centroids = np.zeros((n_samples, self.emb_dim), dtype=np.float32)

            sub_lists = df["requester_subreddits_at_request"].tolist()

            for i, subs in enumerate(sub_lists):
                if not isinstance(subs, list) or not subs:
                    continue

                # Take top K (first K in list)
                current_subs = subs[: self.max_seq_len]

                # Retrieve embeddings
                embs = [sub_to_emb.get(s, dummy_emb) for s in current_subs]
                if not embs:
                    continue

                embs_arr = np.array(embs, dtype=np.float32)
                seq_len = len(embs_arr)

                # Fill sequence tensor
                history_seq[i, :seq_len, :] = embs_arr

                # Fill mask (1 for valid, 0 for padding)
                history_mask[i, :seq_len] = 1.0

                # Compute centroid (mean of valid embeddings)
                centroids[i, :] = np.mean(embs_arr, axis=0)

            # Store in results
            results[split_name] = {
                "title": title_embs,
                "body": body_embs,
                "history": history_seq,
                "history_mask": history_mask,
                "centroid": centroids,
            }

            # Save to cache
            np.savez_compressed(
                self.cache_files[split_name],
                title=title_embs,
                body=body_embs,
                history=history_seq,
                history_mask=history_mask,
                centroid=centroids,
            )

        return results


class TFIDFHandler:
    """
    Handles generation of sparse TF-IDF vectors for the Random Forest model.
    Concatenates Title and Body for a rich text representation.
    """

    def __init__(self):
        self.vocab_size = Config.TFIDF_VOCAB_SIZE
        self.ngram_range = Config.TFIDF_NGRAM_RANGE
        self.cache_dir = Config.CACHE_DIR

        # Cache file paths
        self.cache_files = {
            "train": os.path.join(self.cache_dir, "tfidf_features_train.npz"),
            "val": os.path.join(self.cache_dir, "tfidf_features_val.npz"),
            "test": os.path.join(self.cache_dir, "tfidf_features_test.npz"),
        }

    def _prepare_text(self, df):
        """Concatenates title and body."""
        title = df["request_title"].fillna("").astype(str)
        body = df["request_text_edit_aware"].fillna("").astype(str)
        return title + " " + body

    def process_data(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates or loads TF-IDF features.
        Returns a dictionary containing sparse matrices for 'train', 'val', and 'test'.
        """
        Config.ensure_dirs()

        # Check if all cache files exist
        all_cached = all(os.path.exists(f) for f in self.cache_files.values())

        if load_cached_data and all_cached:
            print("Loading TF-IDF features from cache...")
            results = {}
            for split, path in self.cache_files.items():
                results[split] = sparse.load_npz(path)
            return results

        print("Generating TF-IDF features (Cache miss or force reload)...")

        # Prepare text data
        train_text = self._prepare_text(train_df)
        val_text = self._prepare_text(val_df)
        test_text = self._prepare_text(test_df)

        # Initialize and Fit Vectorizer
        # We only fit on TRAIN data to prevent leakage
        vectorizer = TfidfVectorizer(
            max_features=self.vocab_size,
            ngram_range=self.ngram_range,
            stop_words="english",
            sublinear_tf=True,  # Logarithmic scaling for frequency
        )

        print("Fitting TF-IDF on training data...")
        X_train = vectorizer.fit_transform(train_text)

        print("Transforming validation and test data...")
        X_val = vectorizer.transform(val_text)
        X_test = vectorizer.transform(test_text)

        # Save to cache
        sparse.save_npz(self.cache_files["train"], X_train)
        sparse.save_npz(self.cache_files["val"], X_val)
        sparse.save_npz(self.cache_files["test"], X_test)

        return {"train": X_train, "val": X_val, "test": X_test}
