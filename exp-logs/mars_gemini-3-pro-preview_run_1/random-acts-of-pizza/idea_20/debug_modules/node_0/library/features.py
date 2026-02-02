import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

import library.config as config
from library.data_loader import load_dataset


class FeatureProcessor:
    def __init__(self):
        self.sbert = SentenceTransformer(config.SBERT_MODEL_NAME)
        self.tfidf = TfidfVectorizer(
            max_features=config.TFIDF_VOCAB_SIZE,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.pca = PCA(
            n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_STATE
        )
        self.scaler = StandardScaler()

        # Define numerical columns to use (strictly at_request to avoid leakage)
        self.meta_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

    def _get_text_data(self, df):
        """Concatenates title and body for text processing."""
        # Fill NaNs with empty strings
        title = df["request_title"].fillna("")
        body = df["request_text_edit_aware"].fillna("")
        return (title + " " + body).tolist()

    def _encode_subreddits(self, all_subreddits_list):
        """
        Encodes unique subreddits to save compute.
        Returns a dictionary mapping subreddit name to embedding.
        """
        unique_subs = sorted(
            list(set([sub for sub_list in all_subreddits_list for sub in sub_list]))
        )
        if not unique_subs:
            return {}

        # Encode in batches is handled by SentenceTransformer internally
        embeddings = self.sbert.encode(
            unique_subs, batch_size=64, show_progress_bar=False
        )
        return {sub: emb for sub, emb in zip(unique_subs, embeddings)}

    def _compute_history_features(self, df, sub_emb_map, fit_pca=False):
        """
        Computes:
        1. Centroids (Mean embedding of history)
        2. PCA projection of centroids (Stream A)
        3. Sequences for Attention (Stream B)
        """
        history_lists = df["requester_subreddits_at_request"].tolist()

        centroids = []
        sequences = []
        masks = []

        # Determine max len for this batch (or global max if preferred, here dynamic per batch is easier but we need fixed for MLP)
        # We will use a fixed max length for padding to ensure consistency
        MAX_SEQ_LEN = 50
        embedding_dim = config.SBERT_EMBEDDING_DIM

        for sub_list in history_lists:
            # Filter subs that might not be in map (though they should be if we collected all)
            valid_embs = [sub_emb_map[s] for s in sub_list if s in sub_emb_map]

            # 1. Centroid
            if valid_embs:
                centroid = np.mean(valid_embs, axis=0)
            else:
                centroid = np.zeros(embedding_dim)
            centroids.append(centroid)

            # 3. Sequence & Mask
            # Truncate if too long, pad if too short
            seq_embs = valid_embs[:MAX_SEQ_LEN]
            seq_len = len(seq_embs)

            # Pad
            if seq_len < MAX_SEQ_LEN:
                padding = [
                    np.zeros(embedding_dim) for _ in range(MAX_SEQ_LEN - seq_len)
                ]
                padded_seq = seq_embs + padding
                mask = [1] * seq_len + [0] * (MAX_SEQ_LEN - seq_len)
            else:
                padded_seq = seq_embs
                mask = [1] * MAX_SEQ_LEN

            sequences.append(np.array(padded_seq))
            masks.append(np.array(mask))

        centroids = np.array(centroids)
        sequences = np.array(sequences)  # Shape: (N, 50, 384)
        masks = np.array(masks)  # Shape: (N, 50)

        # 2. PCA Projection
        if fit_pca:
            pca_features = self.pca.fit_transform(centroids)
        else:
            pca_features = self.pca.transform(centroids)

        return centroids, pca_features, sequences, masks

    def _generate_metadata(self, df, fit_scaler=False):
        """
        Generates numerical metadata.
        1. Raw columns
        2. Ratios
        3. Text meta-features
        4. Scaling (for MLP)
        """
        # Extract raw
        meta_df = df[self.meta_cols].copy()

        # Fill NaNs in raw data (simple imputation)
        meta_df = meta_df.fillna(0)

        # Feature Engineering: Ratios
        # Upvote Ratio: up / (up + down) -> derived from (up-down) and (up+down)
        # up+down = total, up-down = diff => 2*up = total+diff => up = (total+diff)/2
        # ratio = up / total
        total_votes = meta_df["requester_upvotes_plus_downvotes_at_request"]
        diff_votes = meta_df["requester_upvotes_minus_downvotes_at_request"]
        # Avoid div by zero
        safe_total = total_votes.replace(0, 1)

        meta_df["upvote_ratio"] = ((total_votes + diff_votes) / 2) / safe_total
        meta_df["comments_per_post"] = meta_df[
            "requester_number_of_comments_at_request"
        ] / (meta_df["requester_number_of_posts_at_request"] + 1)
        meta_df["raop_activity_ratio"] = (
            meta_df["requester_number_of_comments_in_raop_at_request"]
            + meta_df["requester_number_of_posts_on_raop_at_request"]
        ) / (
            meta_df["requester_number_of_comments_at_request"]
            + meta_df["requester_number_of_posts_at_request"]
            + 1
        )

        # Text Meta-Features
        text_data = self._get_text_data(df)
        meta_df["text_len_char"] = [len(t) for t in text_data]
        meta_df["text_len_word"] = [len(t.split()) for t in text_data]
        meta_df["caps_ratio"] = [
            sum(1 for c in t if c.isupper()) / (len(t) + 1) for t in text_data
        ]

        # Convert to numpy
        meta_raw = meta_df.values.astype(np.float32)

        # For MLP: Arcsinh + Standard Scaler
        # Arcsinh handles the heavy tails in karma/comments counts well
        meta_arcsinh = np.arcsinh(meta_raw)

        if fit_scaler:
            meta_scaled = self.scaler.fit_transform(meta_arcsinh)
        else:
            meta_scaled = self.scaler.transform(meta_arcsinh)

        return meta_raw, meta_scaled

    def _compute_consistency_score(self, request_embs, history_centroids):
        """
        Computes Cosine Similarity between Request Embedding and History Centroid.
        """
        # Cosine similarity is dot product of normalized vectors.
        # SBERT embeddings are typically normalized, but we re-normalize to be safe.

        # Norms
        req_norm = np.linalg.norm(request_embs, axis=1, keepdims=True)
        hist_norm = np.linalg.norm(history_centroids, axis=1, keepdims=True)

        # Avoid div by zero
        req_norm[req_norm == 0] = 1e-9
        hist_norm[hist_norm == 0] = 1e-9

        dot_product = np.sum(request_embs * history_centroids, axis=1, keepdims=True)
        similarity = dot_product / (req_norm * hist_norm)

        return similarity

    def process(self, load_cached_data=True):
        """
        Main pipeline execution.
        """
        # Define cache paths
        cache_dir = config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        files = {
            "train_dense": os.path.join(cache_dir, "train_dense.npz"),
            "train_sparse": os.path.join(cache_dir, "train_sparse.npz"),
            "val_dense": os.path.join(cache_dir, "val_dense.npz"),
            "val_sparse": os.path.join(cache_dir, "val_sparse.npz"),
            "test_dense": os.path.join(cache_dir, "test_dense.npz"),
            "test_sparse": os.path.join(cache_dir, "test_sparse.npz"),
        }

        # Check cache
        all_exist = all(os.path.exists(p) for p in files.values())

        if load_cached_data and all_exist:
            print("Loading features from cache...")
            results = {}
            for split in ["train", "val", "test"]:
                dense = np.load(files[f"{split}_dense"])
                sparse = scipy.sparse.load_npz(files[f"{split}_sparse"])

                # Reconstruct dictionary
                results[split] = {
                    "rf_features": sparse,
                    "mlp_request_emb": dense["mlp_request_emb"],
                    "mlp_history_seq": dense["mlp_history_seq"],
                    "mlp_history_mask": dense["mlp_history_mask"],
                    "mlp_metadata": dense["mlp_metadata"],
                }
                if "labels" in dense:
                    results[split]["labels"] = dense["labels"]
            return results["train"], results["val"], results["test"]

        # Compute from scratch
        print("Computing features from scratch...")
        train_df, val_df, test_df = load_dataset(load_cached_data=True)

        # 1. Global Preprocessing (SBERT)
        # Collect all subreddits to encode once
        print("Encoding subreddits...")
        all_subs = (
            train_df["requester_subreddits_at_request"].tolist()
            + val_df["requester_subreddits_at_request"].tolist()
            + test_df["requester_subreddits_at_request"].tolist()
        )
        sub_emb_map = self._encode_subreddits(all_subs)

        # 2. Fit Transformers on Train
        print("Fitting transformers on training data...")
        # TF-IDF
        train_text = self._get_text_data(train_df)
        self.tfidf.fit(train_text)

        # Metadata Scaler & PCA are fitted inside the respective compute methods when fit=True

        # 3. Process Splits
        results = {}
        splits = [("train", train_df), ("val", val_df), ("test", test_df)]

        # We need to compute history centroids first for Train to fit PCA
        print("Processing Train History for PCA...")
        train_centroids, _, _, _ = self._compute_history_features(
            train_df, sub_emb_map, fit_pca=False
        )  # Get raw centroids
        self.pca.fit(train_centroids)  # Fit PCA

        # We need to compute metadata first for Train to fit Scaler
        print("Processing Train Metadata for Scaler...")
        self._generate_metadata(train_df, fit_scaler=True)  # Fits self.scaler

        for name, df in splits:
            print(f"Processing {name} split...")

            # Text Data
            text_list = self._get_text_data(df)

            # --- Stream A: RF Features ---
            # 1. TF-IDF
            tfidf_feats = self.tfidf.transform(text_list)

            # 2. Latent Semantic Centroids & PCA
            centroids, pca_feats, hist_seq, hist_mask = self._compute_history_features(
                df, sub_emb_map, fit_pca=False
            )

            # 3. Request Embeddings (SBERT)
            req_embs = self.sbert.encode(
                text_list, batch_size=64, show_progress_bar=False
            )

            # 4. Consistency Score
            consistency = self._compute_consistency_score(req_embs, centroids)

            # 5. Metadata (Raw for RF, Scaled for MLP)
            meta_raw, meta_scaled = self._generate_metadata(df, fit_scaler=False)

            # Combine RF Features (Sparse)
            # [Meta Raw (Dense), PCA (Dense), Consistency (Dense), TFIDF (Sparse)]
            # Convert dense parts to sparse to hstack
            dense_part = np.hstack([meta_raw, pca_feats, consistency])
            dense_sparse = scipy.sparse.csr_matrix(dense_part)
            rf_features = scipy.sparse.hstack([dense_sparse, tfidf_feats])

            # --- Stream B: MLP Features ---
            # mlp_request_emb = req_embs
            # mlp_history_seq = hist_seq
            # mlp_history_mask = hist_mask
            # mlp_metadata = meta_scaled

            # Store results
            data_dict = {
                "rf_features": rf_features,
                "mlp_request_emb": req_embs,
                "mlp_history_seq": hist_seq,
                "mlp_history_mask": hist_mask,
                "mlp_metadata": meta_scaled,
            }

            # Add labels if available
            if "requester_received_pizza" in df.columns:
                data_dict["labels"] = df["requester_received_pizza"].astype(int).values

            results[name] = data_dict

            # Save to cache
            save_dict = {
                "mlp_request_emb": req_embs,
                "mlp_history_seq": hist_seq,
                "mlp_history_mask": hist_mask,
                "mlp_metadata": meta_scaled,
            }
            if "labels" in data_dict:
                save_dict["labels"] = data_dict["labels"]

            np.savez(files[f"{name}_dense"], **save_dict)
            scipy.sparse.save_npz(files[f"{name}_sparse"], rf_features)

        return results["train"], results["val"], results["test"]
