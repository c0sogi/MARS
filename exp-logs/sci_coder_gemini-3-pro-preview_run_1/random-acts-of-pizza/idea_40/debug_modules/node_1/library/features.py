import os
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from scipy import sparse
from library import config, utils


class FeatureEngineer:
    """
    Implements the feature engineering pipeline for the Hybrid Ensemble model.
    Generates features for two streams:
    1. Stream A (Random Forest): TF-IDF + Metadata + Top-K Indicators + Consistency Scalars
    2. Stream B (MLP): SBERT Embeddings (Title, Body, History Sequence) + Scaled Metadata + Consistency Scalars
    """

    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        self.sbert_model_name = config.SBERT_MODEL_NAME
        self.tfidf_max_features = config.TFIDF_MAX_FEATURES
        self.top_k = config.TOP_K_SUBREDDITS
        self.max_history_len = 20  # Max sequence length for user history in MLP

        # Cache file paths
        self.rf_cache_path = os.path.join(self.cache_dir, "rf_features.npz")
        self.mlp_cache_path = os.path.join(self.cache_dir, "mlp_features.npz")

    def _parse_subreddits(self, df):
        """Parses the stringified list of subreddits."""
        return df["requester_subreddits_at_request"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )

    def _get_sbert_embeddings(self, texts, model):
        """Generates SBERT embeddings for a list of texts."""
        # Handle empty or non-string inputs
        cleaned_texts = [t if isinstance(t, str) and len(t) > 0 else " " for t in texts]
        return model.encode(cleaned_texts, batch_size=64, show_progress_bar=False)

    def _compute_history_features(self, df_train, df_val, df_test, sbert_model):
        """
        Computes history-related features:
        1. Subreddit Embeddings & User Centroids
        2. Top-K Binary Indicators
        3. Padded History Sequences for MLP
        """
        # Parse subreddits
        train_subs = self._parse_subreddits(df_train)
        val_subs = self._parse_subreddits(df_val)
        test_subs = self._parse_subreddits(df_test)

        all_subs_lists = pd.concat([train_subs, val_subs, test_subs], axis=0)

        # 1. Identify Unique Subreddits and Embed Them
        unique_subreddits = sorted(
            list(set([item for sublist in all_subs_lists for item in sublist]))
        )
        # Create a mapping from subreddit name to index and embedding
        sub_to_idx = {sub: i for i, sub in enumerate(unique_subreddits)}

        # Embed unique subreddits (batch processing)
        if unique_subreddits:
            sub_embeddings = sbert_model.encode(
                unique_subreddits, batch_size=128, show_progress_bar=False
            )
        else:
            sub_embeddings = np.zeros((0, 384))

        # 2. Top-K Subreddits (based on Train frequency)
        flat_train_subs = [item for sublist in train_subs for item in sublist]
        sub_counts = pd.Series(flat_train_subs).value_counts()
        top_k_subs = sub_counts.head(self.top_k).index.tolist()
        top_k_mapping = {sub: i for i, sub in enumerate(top_k_subs)}

        def get_top_k_matrix(subs_series):
            matrix = np.zeros((len(subs_series), self.top_k), dtype=np.float32)
            for i, sub_list in enumerate(subs_series):
                for sub in sub_list:
                    if sub in top_k_mapping:
                        matrix[i, top_k_mapping[sub]] = 1.0
            return matrix

        train_top_k = get_top_k_matrix(train_subs)
        val_top_k = get_top_k_matrix(val_subs)
        test_top_k = get_top_k_matrix(test_subs)

        # 3. Compute Centroids and History Sequences
        def process_history(subs_series):
            n_samples = len(subs_series)
            centroids = np.zeros((n_samples, 384), dtype=np.float32)
            sequences = np.zeros(
                (n_samples, self.max_history_len, 384), dtype=np.float32
            )
            masks = np.zeros(
                (n_samples, self.max_history_len), dtype=np.float32
            )  # 1 for valid, 0 for pad

            for i, sub_list in enumerate(subs_series):
                if not sub_list:
                    continue

                # Get indices for unique subreddits
                indices = [sub_to_idx[s] for s in sub_list if s in sub_to_idx]
                if not indices:
                    continue

                # Get embeddings
                curr_embs = sub_embeddings[indices]

                # Centroid
                centroids[i] = np.mean(curr_embs, axis=0)

                # Sequence (Truncate or Pad)
                seq_len = min(len(curr_embs), self.max_history_len)
                sequences[i, :seq_len, :] = curr_embs[:seq_len]
                masks[i, :seq_len] = 1.0

            return centroids, sequences, masks

        train_centroids, train_seq, train_mask = process_history(train_subs)
        val_centroids, val_seq, val_mask = process_history(val_subs)
        test_centroids, test_seq, test_mask = process_history(test_subs)

        return (
            (train_top_k, val_top_k, test_top_k),
            (train_centroids, val_centroids, test_centroids),
            (train_seq, val_seq, test_seq),
            (train_mask, val_mask, test_mask),
        )

    def process_data(self, load_cached_data=True):
        """
        Main execution method.
        If cached data exists and load_cached_data is True, returns loaded data.
        Otherwise, computes features from scratch and saves them.
        """
        # 1. Check Cache
        if (
            load_cached_data
            and os.path.exists(self.rf_cache_path)
            and os.path.exists(self.mlp_cache_path)
        ):
            print("Loading features from cache...")
            rf_data = np.load(self.rf_cache_path, allow_pickle=True)
            mlp_data = np.load(self.mlp_cache_path, allow_pickle=True)

            # Reconstruct dictionaries
            rf_out = {k: rf_data[k] for k in rf_data.files}
            # Sparse matrices are stored as objects in npz usually, but here we likely saved dense or handled sparse separately.
            # For simplicity in this pipeline, we will densify TF-IDF if it's not too huge, or rely on the fact that
            # we are concatenating everything.
            # However, np.savez converts sparse to 0-d object array if not careful.
            # To ensure robustness, we will assume the cache stores final concatenated dense arrays or
            # we accept re-computation if strict sparse handling is needed.
            # Given 5000 features + metadata, dense is manageable (5000 * 4000 * 4 bytes ~ 80MB).

            mlp_out = {k: mlp_data[k] for k in mlp_data.files}
            return rf_out, mlp_out

        print("Computing features from scratch...")

        # 2. Load Raw Data
        df_train, df_val, df_test = utils.load_data()

        # Extract Targets
        y_train = df_train["requester_received_pizza"].astype(int).values
        y_val = df_val["requester_received_pizza"].astype(int).values
        # Test target is not used for prediction, but kept for structure if needed (usually 0s)

        # 3. Text Processing (SBERT)
        print("Generating SBERT embeddings...")
        sbert = SentenceTransformer(self.sbert_model_name)

        # Title Embeddings
        train_title_emb = self._get_sbert_embeddings(df_train["request_title"], sbert)
        val_title_emb = self._get_sbert_embeddings(df_val["request_title"], sbert)
        test_title_emb = self._get_sbert_embeddings(df_test["request_title"], sbert)

        # Body Embeddings (Use edit aware)
        train_body_emb = self._get_sbert_embeddings(
            df_train["request_text_edit_aware"], sbert
        )
        val_body_emb = self._get_sbert_embeddings(
            df_val["request_text_edit_aware"], sbert
        )
        test_body_emb = self._get_sbert_embeddings(
            df_test["request_text_edit_aware"], sbert
        )

        # 4. History Features
        print("Processing user history...")
        (top_k_data, centroids_data, seq_data, mask_data) = (
            self._compute_history_features(df_train, df_val, df_test, sbert)
        )
        train_top_k, val_top_k, test_top_k = top_k_data
        train_centroids, val_centroids, test_centroids = centroids_data
        train_seq, val_seq, test_seq = seq_data
        train_mask, val_mask, test_mask = mask_data

        # 5. Consistency Scalars (Cosine Similarity)
        # Title vs History
        def get_cosine(a, b):
            # a: (N, D), b: (N, D) -> returns (N, 1)
            sim = np.sum(a * b, axis=1) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
            )
            return sim.reshape(-1, 1)

        train_cons_title = get_cosine(train_title_emb, train_centroids)
        val_cons_title = get_cosine(val_title_emb, val_centroids)
        test_cons_title = get_cosine(test_title_emb, test_centroids)

        # Body vs History
        train_cons_body = get_cosine(train_body_emb, train_centroids)
        val_cons_body = get_cosine(val_body_emb, val_centroids)
        test_cons_body = get_cosine(test_body_emb, test_centroids)

        # Concatenate Consistency
        train_consistency = np.hstack([train_cons_title, train_cons_body])
        val_consistency = np.hstack([val_cons_title, val_cons_body])
        test_consistency = np.hstack([test_cons_title, test_cons_body])

        # 6. Metadata Processing
        print("Processing metadata...")
        # Identify common numerical columns
        common_cols = utils.get_common_columns(df_train, df_test)
        # Filter for numeric only
        numeric_cols = (
            df_train[common_cols].select_dtypes(include=[np.number]).columns.tolist()
        )

        # Extract raw metadata
        X_meta_train_raw = df_train[numeric_cols].values
        X_meta_val_raw = df_val[numeric_cols].values
        X_meta_test_raw = df_test[numeric_cols].values

        # Impute (Median)
        imputer = SimpleImputer(strategy="median")
        X_meta_train_imp = imputer.fit_transform(X_meta_train_raw)
        X_meta_val_imp = imputer.transform(X_meta_val_raw)
        X_meta_test_imp = imputer.transform(X_meta_test_raw)

        # --- Stream A (RF) Specifics ---
        # TF-IDF
        print("Generating TF-IDF...")
        tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features, stop_words="english"
        )
        # Combine title and body for TF-IDF
        train_text = (
            df_train["request_title"].fillna("")
            + " "
            + df_train["request_text_edit_aware"].fillna("")
        )
        val_text = (
            df_val["request_title"].fillna("")
            + " "
            + df_val["request_text_edit_aware"].fillna("")
        )
        test_text = (
            df_test["request_title"].fillna("")
            + " "
            + df_test["request_text_edit_aware"].fillna("")
        )

        X_tfidf_train = tfidf.fit_transform(train_text).toarray()
        X_tfidf_val = tfidf.transform(val_text).toarray()
        X_tfidf_test = tfidf.transform(test_text).toarray()

        # Assemble RF Features
        # [TF-IDF, Metadata, Top-K, Consistency]
        X_train_rf = np.hstack(
            [X_tfidf_train, X_meta_train_imp, train_top_k, train_consistency]
        )
        X_val_rf = np.hstack([X_tfidf_val, X_meta_val_imp, val_top_k, val_consistency])
        X_test_rf = np.hstack(
            [X_tfidf_test, X_meta_test_imp, test_top_k, test_consistency]
        )

        rf_out = {
            "X_train": X_train_rf,
            "y_train": y_train,
            "X_val": X_val_rf,
            "y_val": y_val,
            "X_test": X_test_rf,
            "request_ids_test": df_test["request_id"].values,
        }

        # --- Stream B (MLP) Specifics ---
        # Metadata: Arcsinh -> StandardScale
        X_meta_train_arcsinh = np.arcsinh(X_meta_train_imp)
        X_meta_val_arcsinh = np.arcsinh(X_meta_val_imp)
        X_meta_test_arcsinh = np.arcsinh(X_meta_test_imp)

        scaler = StandardScaler()
        X_meta_train_scaled = scaler.fit_transform(X_meta_train_arcsinh)
        X_meta_val_scaled = scaler.transform(X_meta_val_arcsinh)
        X_meta_test_scaled = scaler.transform(X_meta_test_arcsinh)

        mlp_out = {
            # Embeddings
            "train_title_emb": train_title_emb,
            "train_body_emb": train_body_emb,
            "val_title_emb": val_title_emb,
            "val_body_emb": val_body_emb,
            "test_title_emb": test_title_emb,
            "test_body_emb": test_body_emb,
            # History Sequences & Masks
            "train_hist_seq": train_seq,
            "train_hist_mask": train_mask,
            "val_hist_seq": val_seq,
            "val_hist_mask": val_mask,
            "test_hist_seq": test_seq,
            "test_hist_mask": test_mask,
            # Metadata & Consistency
            "train_meta": X_meta_train_scaled,
            "train_cons": train_consistency,
            "val_meta": X_meta_val_scaled,
            "val_cons": val_consistency,
            "test_meta": X_meta_test_scaled,
            "test_cons": test_consistency,
            # Targets
            "y_train": y_train,
            "y_val": y_val,
            "request_ids_test": df_test["request_id"].values,
        }

        # 7. Save to Cache
        print("Saving features to cache...")
        np.savez(self.rf_cache_path, **rf_out)
        np.savez(self.mlp_cache_path, **mlp_out)

        return rf_out, mlp_out
