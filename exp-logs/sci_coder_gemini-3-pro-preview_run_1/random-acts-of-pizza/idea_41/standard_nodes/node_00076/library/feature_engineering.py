import os
import ast
import numpy as np
import pandas as pd
import scipy.sparse as sp
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from library.config import (
    WORKING_DIR,
    NUMERIC_FEATURES,
    TFIDF_MAX_FEATURES,
    TOP_K_COMMUNITIES,
    RANDOM_STATE,
    CACHE_PATHS,
)
from library.embedding_utils import SBERTEmbedder
from library.utils import set_seed

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception:
        pass  # Handle gracefully if download fails


class FeaturePipeline:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.top_k_subreddits = []
        self.embedder = SBERTEmbedder()
        self.sia = SentimentIntensityAnalyzer()

        # Max sequence length for user history in MLP
        self.max_history_len = 50

    def _parse_subreddits(self, df):
        """Parses the subreddit list column from string to list."""
        col = "requester_subreddits_at_request"
        if col in df.columns:
            # Handle if it's already a list or needs parsing
            return df[col].apply(
                lambda x: (
                    ast.literal_eval(x)
                    if isinstance(x, str)
                    else (x if isinstance(x, list) else [])
                )
            )
        return pd.Series([[]] * len(df))

    def _extract_metadata(self, df):
        """Extracts and engineers numerical and text meta-features."""
        # 1. Base Numerical Features
        features = df[NUMERIC_FEATURES].copy()

        # 2. Ratios and Interactions
        # Avoid division by zero
        up = df.get("requester_upvotes_minus_downvotes_at_request", 0)
        total = df.get("requester_upvotes_plus_downvotes_at_request", 0)
        features["upvote_ratio"] = np.where(total > 0, (total + up) / (2 * total), 0.5)

        # 3. Text Meta-Features
        for col in ["request_title", "request_text_edit_aware"]:
            text_col = df[col].fillna("").astype(str)
            prefix = col.replace("request_", "")
            features[f"{prefix}_len_char"] = text_col.apply(len)
            features[f"{prefix}_len_word"] = text_col.apply(lambda x: len(x.split()))
            features[f"{prefix}_caps_ratio"] = text_col.apply(
                lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
            )

            # Sentiment
            features[f"{prefix}_sentiment"] = text_col.apply(
                lambda x: self.sia.polarity_scores(x)["compound"]
            )

        return features

    def _extract_top_k_communities(self, train_subs, val_subs, test_subs):
        """Generates binary flags for top-K subreddits."""
        # Flatten train subreddits to count frequencies
        all_subs = [sub for sub_list in train_subs for sub in sub_list]
        counts = pd.Series(all_subs).value_counts()
        self.top_k_subreddits = counts.head(TOP_K_COMMUNITIES).index.tolist()

        def get_flags(sub_series):
            # Create a matrix of shape (n_samples, k)
            flags = np.zeros((len(sub_series), len(self.top_k_subreddits)), dtype=int)
            for i, subs in enumerate(sub_series):
                s_set = set(subs)
                for j, target in enumerate(self.top_k_subreddits):
                    if target in s_set:
                        flags[i, j] = 1
            return flags

        return (get_flags(train_subs), get_flags(val_subs), get_flags(test_subs))

    def _compute_embeddings_and_consistency(self, df, sub_lists, dataset_name):
        """
        Computes SBERT embeddings for Title, Body, and History.
        Calculates User Centroid and Consistency Scalars.
        """
        # 1. Title and Body Embeddings
        titles = df["request_title"].fillna("").astype(str).tolist()
        bodies = df["request_text_edit_aware"].fillna("").astype(str).tolist()

        title_embs = self.embedder.encode_batch(
            titles, cache_key=f"title_emb_{dataset_name}"
        )
        body_embs = self.embedder.encode_batch(
            bodies, cache_key=f"body_emb_{dataset_name}"
        )

        # 2. History Embeddings
        # Identify all unique subreddits in this split to batch encode
        unique_subs = list(set(sub for sub_list in sub_lists for sub in sub_list))
        if unique_subs:
            sub_embs_map = dict(
                zip(
                    unique_subs,
                    self.embedder.encode_batch(
                        unique_subs, cache_key=f"sub_emb_map_{dataset_name}"
                    ),
                )
            )
        else:
            sub_embs_map = {}

        # 3. Construct Sequences and Centroids
        n_samples = len(df)
        emb_dim = 384

        history_seqs = np.zeros(
            (n_samples, self.max_history_len, emb_dim), dtype=np.float32
        )
        history_masks = np.zeros((n_samples, self.max_history_len), dtype=np.float32)
        centroids = np.zeros((n_samples, emb_dim), dtype=np.float32)

        for i, subs in enumerate(sub_lists):
            # Truncate or pad
            valid_subs = [s for s in subs if s in sub_embs_map][: self.max_history_len]

            if valid_subs:
                # Stack embeddings for this user
                user_sub_embs = np.stack([sub_embs_map[s] for s in valid_subs])

                # Sequence
                seq_len = len(user_sub_embs)
                history_seqs[i, :seq_len, :] = user_sub_embs
                history_masks[i, :seq_len] = 1.0

                # Centroid (Global Persona)
                centroids[i] = np.mean(user_sub_embs, axis=0)
            else:
                # No history: Zero vector centroid (neutral)
                pass

        # 4. Consistency Scalars (Cosine Similarity)
        # Reshape for broadcasting if needed, but here simple dot product of normalized vectors
        # SBERT embeddings are normalized, so dot product is cosine similarity

        # Ensure normalization just in case
        def normalize(v):
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            return np.divide(v, norm, out=np.zeros_like(v), where=norm != 0)

        title_embs_norm = normalize(title_embs)
        body_embs_norm = normalize(body_embs)
        centroids_norm = normalize(centroids)

        # Dot product
        title_consistency = np.sum(
            title_embs_norm * centroids_norm, axis=1, keepdims=True
        )
        body_consistency = np.sum(
            body_embs_norm * centroids_norm, axis=1, keepdims=True
        )

        consistency_scalars = np.hstack([title_consistency, body_consistency])

        return {
            "title_emb": title_embs,
            "body_emb": body_embs,
            "history_emb": history_seqs,
            "history_mask": history_masks,
            "history_centroid": centroids,
            "consistency_scalars": consistency_scalars,
        }

    def fit_transform(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main execution method.
        Checks cache, otherwise computes features and saves them.
        """
        # Define cache files
        cache_files = [
            CACHE_PATHS["rf_train"],
            CACHE_PATHS["rf_val"],
            CACHE_PATHS["rf_test"],
            CACHE_PATHS["mlp_train"],
            CACHE_PATHS["mlp_val"],
            CACHE_PATHS["mlp_test"],
        ]

        # Check if all cache files exist
        if load_cached_data and all(os.path.exists(f) for f in cache_files):
            print("Loading features from cache...")
            results = {}
            for split in ["train", "val", "test"]:
                # Load RF (sparse or npz)
                rf_data = np.load(CACHE_PATHS[f"rf_{split}"], allow_pickle=True)
                # Handle sparse matrix stored in npz
                if "data" in rf_data.files:
                    # Reconstruct sparse matrix
                    X_rf = sp.csr_matrix(
                        (rf_data["data"], rf_data["indices"], rf_data["indptr"]),
                        shape=rf_data["shape"],
                    )
                else:
                    # Fallback if stored differently
                    X_rf = rf_data["arr_0"] if "arr_0" in rf_data else None

                # Load MLP
                mlp_data = np.load(CACHE_PATHS[f"mlp_{split}"], allow_pickle=True)
                X_mlp = {k: mlp_data[k] for k in mlp_data.files}

                results[split] = {"rf": X_rf, "mlp": X_mlp}
            return results

        print("Computing features from scratch...")

        # 1. Parse Subreddits
        train_subs = self._parse_subreddits(train_df)
        val_subs = self._parse_subreddits(val_df)
        test_subs = self._parse_subreddits(test_df)

        # 2. Metadata Engineering
        meta_train = self._extract_metadata(train_df)
        meta_val = self._extract_metadata(val_df)
        meta_test = self._extract_metadata(test_df)

        # Impute for RF (Simple Median)
        meta_train_imp = self.imputer.fit_transform(meta_train)
        meta_val_imp = self.imputer.transform(meta_val)
        meta_test_imp = self.imputer.transform(meta_test)

        # Scale for MLP (Arcsinh + Standard)
        # Arcsinh handles skewed count data well
        transformer = FunctionTransformer(np.arcsinh)
        meta_train_log = transformer.transform(meta_train_imp)
        meta_val_log = transformer.transform(meta_val_imp)
        meta_test_log = transformer.transform(meta_test_imp)

        meta_train_scaled = self.scaler.fit_transform(meta_train_log)
        meta_val_scaled = self.scaler.transform(meta_val_log)
        meta_test_scaled = self.scaler.transform(meta_test_log)

        # 3. Top-K Communities
        topk_train, topk_val, topk_test = self._extract_top_k_communities(
            train_subs, val_subs, test_subs
        )

        # 4. TF-IDF (Stream A)
        # Concat title and body
        def get_text(df):
            return (
                df["request_title"].fillna("")
                + " "
                + df["request_text_edit_aware"].fillna("")
            ).astype(str)

        tfidf_train = self.tfidf.fit_transform(get_text(train_df))
        tfidf_val = self.tfidf.transform(get_text(val_df))
        tfidf_test = self.tfidf.transform(get_text(test_df))

        # 5. Embeddings & Consistency (Stream B + Stream A scalars)
        emb_train = self._compute_embeddings_and_consistency(
            train_df, train_subs, "train"
        )
        emb_val = self._compute_embeddings_and_consistency(val_df, val_subs, "val")
        emb_test = self._compute_embeddings_and_consistency(test_df, test_subs, "test")

        # 6. Assembly
        results = {}

        for split, meta_imp, meta_sc, topk, tfidf, emb, cache_rf, cache_mlp in zip(
            ["train", "val", "test"],
            [meta_train_imp, meta_val_imp, meta_test_imp],
            [meta_train_scaled, meta_val_scaled, meta_test_scaled],
            [topk_train, topk_val, topk_test],
            [tfidf_train, tfidf_val, tfidf_test],
            [emb_train, emb_val, emb_test],
            [CACHE_PATHS["rf_train"], CACHE_PATHS["rf_val"], CACHE_PATHS["rf_test"]],
            [CACHE_PATHS["mlp_train"], CACHE_PATHS["mlp_val"], CACHE_PATHS["mlp_test"]],
        ):
            # Stream A: RF Features
            # [TFIDF (Sparse) | Metadata (Dense) | TopK (Dense) | Consistency (Dense)]
            # Convert dense parts to sparse to stack efficiently
            dense_part = np.hstack([meta_imp, topk, emb["consistency_scalars"]])
            X_rf = sp.hstack([tfidf, sp.csr_matrix(dense_part)])

            # Stream B: MLP Features
            # Dense features for MLP = Scaled Metadata + TopK + Consistency
            dense_mlp = np.hstack([meta_sc, topk, emb["consistency_scalars"]]).astype(
                np.float32
            )

            X_mlp = {
                "title_emb": emb["title_emb"].astype(np.float32),
                "body_emb": emb["body_emb"].astype(np.float32),
                "history_emb": emb["history_emb"],
                "history_mask": emb["history_mask"],
                "history_centroid": emb["history_centroid"],
                "dense_features": dense_mlp,
            }

            results[split] = {"rf": X_rf, "mlp": X_mlp}

            # Save to Cache
            # Save RF as sparse components to npz
            sp.save_npz(
                cache_rf.replace(".npz", "_sparse.npz"), X_rf
            )  # Backup standard sparse save
            # Custom save for consistency with load logic above
            np.savez(
                cache_rf,
                data=X_rf.data,
                indices=X_rf.indices,
                indptr=X_rf.indptr,
                shape=X_rf.shape,
            )

            # Save MLP
            np.savez(cache_mlp, **X_mlp)

        return results
