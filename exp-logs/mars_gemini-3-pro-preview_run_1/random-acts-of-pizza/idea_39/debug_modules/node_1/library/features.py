import os
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import configuration and utilities
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    SBERT_MODEL_NAME,
    TFIDF_VOCAB_SIZE,
    TOP_K_COMMUNITIES,
    SEED,
    DEBUG_MODE,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import set_seed


class FeaturePipeline:
    def __init__(self):
        set_seed(SEED)
        self.cache_dir = CACHE_DIR
        self.rf_features_path = os.path.join(self.cache_dir, "rf_features.npz")
        self.mlp_features_path = os.path.join(self.cache_dir, "mlp_features.npz")

        # Initialize processors
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_VOCAB_SIZE,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")

        # SBERT model is loaded lazily or during processing to save memory if cached
        self.sbert_model = None

    def _load_raw_data(self):
        """Loads raw CSVs from metadata directory."""
        print("Loading raw data from metadata...")
        converters = {"requester_subreddits_at_request": ast.literal_eval}

        df_train = pd.read_csv(TRAIN_PATH, converters=converters)
        df_val = pd.read_csv(VAL_PATH, converters=converters)
        df_test = pd.read_csv(TEST_PATH, converters=converters)

        if DEBUG_MODE:
            print(f"DEBUG MODE: Subsampling {DEBUG_SAMPLE_SIZE} rows.")
            df_train = df_train.iloc[:DEBUG_SAMPLE_SIZE]
            df_val = df_val.iloc[:DEBUG_SAMPLE_SIZE]
            df_test = df_test.iloc[:DEBUG_SAMPLE_SIZE]

        return df_train, df_val, df_test

    def _get_sbert_model(self):
        if self.sbert_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading SBERT model {SBERT_MODEL_NAME} on {device}...")
            self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
        return self.sbert_model

    def _encode_text_sbert(self, text_list, desc="Encoding"):
        """Encodes a list of texts using SBERT."""
        model = self._get_sbert_model()
        # Replace NaNs or empty strings
        text_list = [str(t) if pd.notna(t) and t != "" else " " for t in text_list]
        embeddings = model.encode(
            text_list,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def _process_history_embeddings(self, df_list):
        """
        Generates embeddings for user history.
        Returns:
            - history_sequences: List of arrays (Seq_Len, Emb_Dim)
            - history_centroids: Array (N, Emb_Dim)
        """
        model = self._get_sbert_model()

        # Flatten all unique subreddits to encode efficiently
        all_subreddits = set()
        for sub_list in df_list:
            for subs in sub_list:
                all_subreddits.update(subs)

        unique_subs = list(all_subreddits)
        if not unique_subs:
            # Handle edge case where no subreddits exist in entire dataset
            sub_to_emb = {}
            emb_dim = model.get_sentence_embedding_dimension()
        else:
            print(f"Encoding {len(unique_subs)} unique subreddits...")
            sub_embeddings = model.encode(
                unique_subs,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            sub_to_emb = {sub: emb for sub, emb in zip(unique_subs, sub_embeddings)}
            emb_dim = sub_embeddings.shape[1]

        # Reconstruct per-user history
        history_sequences = []
        history_centroids = []

        print("Constructing user history sequences and centroids...")
        for subs in tqdm(df_list):
            if not subs:
                # No history
                history_sequences.append(np.zeros((0, emb_dim), dtype=np.float32))
                history_centroids.append(np.zeros(emb_dim, dtype=np.float32))
            else:
                embs = np.array([sub_to_emb[s] for s in subs if s in sub_to_emb])
                if len(embs) == 0:
                    history_sequences.append(np.zeros((0, emb_dim), dtype=np.float32))
                    history_centroids.append(np.zeros(emb_dim, dtype=np.float32))
                else:
                    history_sequences.append(embs)
                    history_centroids.append(np.mean(embs, axis=0))

        return history_sequences, np.array(history_centroids)

    def _pad_sequences(self, sequences, max_len=50):
        """Pads sequences to fixed length for MLP."""
        N = len(sequences)
        if N == 0:
            return np.zeros((0, max_len, 384), dtype=np.float32)

        emb_dim = 384  # SBERT default
        if len(sequences) > 0 and len(sequences[0]) > 0:
            emb_dim = sequences[0].shape[1]
        elif hasattr(self, "sbert_model") and self.sbert_model:
            emb_dim = self.sbert_model.get_sentence_embedding_dimension()

        padded = np.zeros((N, max_len, emb_dim), dtype=np.float32)

        for i, seq in enumerate(sequences):
            if len(seq) > 0:
                length = min(len(seq), max_len)
                # Take most recent (assuming list is chronological or just take first K)
                # The dataset doesn't strictly specify order, usually sorted by time or frequency.
                # We'll take the first K provided in the list.
                padded[i, :length, :] = seq[:length]

        return padded

    def _extract_metadata_and_topk(self, df_train, df_val, df_test):
        """
        Extracts numerical metadata and Top-K community flags.
        """
        print("Extracting metadata and Top-K communities...")

        # 1. Top-K Communities
        # Count frequencies in train
        subreddit_counts = {}
        for subs in df_train["requester_subreddits_at_request"]:
            for sub in subs:
                subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1

        sorted_subs = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)
        top_k_subs = [s[0] for s in sorted_subs[:TOP_K_COMMUNITIES]]

        def get_topk_features(df):
            features = np.zeros((len(df), TOP_K_COMMUNITIES), dtype=np.float32)
            for i, subs in enumerate(df["requester_subreddits_at_request"]):
                sub_set = set(subs)
                for j, target_sub in enumerate(top_k_subs):
                    if target_sub in sub_set:
                        features[i, j] = 1.0
            return features

        topk_train = get_topk_features(df_train)
        topk_val = get_topk_features(df_val)
        topk_test = get_topk_features(df_test)

        # 2. Numerical Metadata
        # Define columns to extract
        num_cols = [
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

        def get_numeric_features(df):
            # Extract raw
            data = df[num_cols].copy()

            # Fill NaNs with 0 for these counts/ages
            data = data.fillna(0)

            # Feature Engineering: Ratios and Text Stats
            # Upvote Ratio (avoid div by zero)
            total_votes = data["requester_upvotes_plus_downvotes_at_request"]
            net_votes = data["requester_upvotes_minus_downvotes_at_request"]
            # approx upvotes = (total + net) / 2
            upvotes = (total_votes + net_votes) / 2
            data["upvote_ratio"] = np.where(total_votes > 0, upvotes / total_votes, 0.5)

            # Text Lengths
            data["title_len_char"] = df["request_title"].fillna("").apply(len)
            data["body_len_char"] = df["request_text_edit_aware"].fillna("").apply(len)
            data["title_len_word"] = (
                df["request_title"].fillna("").apply(lambda x: len(str(x).split()))
            )
            data["body_len_word"] = (
                df["request_text_edit_aware"]
                .fillna("")
                .apply(lambda x: len(str(x).split()))
            )

            return data.values.astype(np.float32)

        meta_train = get_numeric_features(df_train)
        meta_val = get_numeric_features(df_val)
        meta_test = get_numeric_features(df_test)

        # 3. Concatenate (Meta + TopK)
        # For RF: We use this directly (handled later with imputation/scaling)
        # For MLP: We need to scale the numeric part specifically

        return (meta_train, topk_train), (meta_val, topk_val), (meta_test, topk_test)

    def run(self, load_cached_data=True):
        """
        Executes the pipeline.
        Returns:
            rf_data (dict): Data for Random Forest
            mlp_data (dict): Data for MLP
        """
        # Check cache
        if (
            load_cached_data
            and os.path.exists(self.rf_features_path)
            and os.path.exists(self.mlp_features_path)
        ):
            print("Loading features from cache...")
            rf_data = np.load(self.rf_features_path, allow_pickle=True)
            mlp_data = np.load(self.mlp_features_path, allow_pickle=True)
            # Convert np.load result to dict to avoid pickling issues if closed
            return dict(rf_data), dict(mlp_data)

        # Load Data
        df_train, df_val, df_test = self._load_raw_data()

        # Targets
        y_train = df_train["requester_received_pizza"].astype(int).values
        y_val = df_val["requester_received_pizza"].astype(int).values
        # Test has no target for prediction, but we handle it structurally

        # ==========================
        # 1. Text Embeddings (SBERT)
        # ==========================
        print("Generating SBERT embeddings...")
        train_title_emb = self._encode_text_sbert(df_train["request_title"].tolist())
        val_title_emb = self._encode_text_sbert(df_val["request_title"].tolist())
        test_title_emb = self._encode_text_sbert(df_test["request_title"].tolist())

        train_body_emb = self._encode_text_sbert(
            df_train["request_text_edit_aware"].tolist()
        )
        val_body_emb = self._encode_text_sbert(
            df_val["request_text_edit_aware"].tolist()
        )
        test_body_emb = self._encode_text_sbert(
            df_test["request_text_edit_aware"].tolist()
        )

        # History Embeddings
        print("Processing User History...")
        # Concatenate lists to process unique subs once (optimization inside _process_history_embeddings)
        all_dfs_subs = (
            df_train["requester_subreddits_at_request"].tolist()
            + df_val["requester_subreddits_at_request"].tolist()
            + df_test["requester_subreddits_at_request"].tolist()
        )

        all_hist_seqs, all_hist_centroids = self._process_history_embeddings(
            all_dfs_subs
        )

        # Split back
        n_train = len(df_train)
        n_val = len(df_val)

        train_hist_seqs = all_hist_seqs[:n_train]
        val_hist_seqs = all_hist_seqs[n_train : n_train + n_val]
        test_hist_seqs = all_hist_seqs[n_train + n_val :]

        train_hist_cen = all_hist_centroids[:n_train]
        val_hist_cen = all_hist_centroids[n_train : n_train + n_val]
        test_hist_cen = all_hist_centroids[n_train + n_val :]

        # ==========================
        # 2. Metadata & Top-K
        # ==========================
        (
            (meta_train_raw, topk_train),
            (meta_val_raw, topk_val),
            (meta_test_raw, topk_test),
        ) = self._extract_metadata_and_topk(df_train, df_val, df_test)

        # ==========================
        # 3. Feature Engineering for RF (Stream A)
        # ==========================
        print("Preparing RF features...")

        # TF-IDF
        full_text_train = (
            df_train["request_title"].fillna("")
            + " "
            + df_train["request_text_edit_aware"].fillna("")
        )
        full_text_val = (
            df_val["request_title"].fillna("")
            + " "
            + df_val["request_text_edit_aware"].fillna("")
        )
        full_text_test = (
            df_test["request_title"].fillna("")
            + " "
            + df_test["request_text_edit_aware"].fillna("")
        )

        tfidf_train = self.tfidf.fit_transform(full_text_train).toarray()
        tfidf_val = self.tfidf.transform(full_text_val).toarray()
        tfidf_test = self.tfidf.transform(full_text_test).toarray()

        # Consistency Scalars (Cosine Sim)
        def get_consistency(emb_text, emb_hist):
            # Dot product of normalized vectors is cosine similarity
            # SBERT output is already normalized if normalize_embeddings=True
            # Centroids might not be normalized after averaging
            norm_hist = np.linalg.norm(emb_hist, axis=1, keepdims=True)
            norm_hist[norm_hist == 0] = 1.0  # Avoid div by zero
            emb_hist_norm = emb_hist / norm_hist

            # Row-wise dot product
            sim = np.sum(emb_text * emb_hist_norm, axis=1, keepdims=True)
            return sim

        train_cons_title = get_consistency(train_title_emb, train_hist_cen)
        val_cons_title = get_consistency(val_title_emb, val_hist_cen)
        test_cons_title = get_consistency(test_title_emb, test_hist_cen)

        train_cons_body = get_consistency(train_body_emb, train_hist_cen)
        val_cons_body = get_consistency(val_body_emb, val_hist_cen)
        test_cons_body = get_consistency(test_body_emb, test_hist_cen)

        # Impute Metadata for RF
        self.imputer.fit(meta_train_raw)
        meta_train_imp = self.imputer.transform(meta_train_raw)
        meta_val_imp = self.imputer.transform(meta_val_raw)
        meta_test_imp = self.imputer.transform(meta_test_raw)

        # Assemble RF Matrix: [TFIDF, Meta, TopK, Consistency]
        X_train_rf = np.hstack(
            [tfidf_train, meta_train_imp, topk_train, train_cons_title, train_cons_body]
        )
        X_val_rf = np.hstack(
            [tfidf_val, meta_val_imp, topk_val, val_cons_title, val_cons_body]
        )
        X_test_rf = np.hstack(
            [tfidf_test, meta_test_imp, topk_test, test_cons_title, test_cons_body]
        )

        # ==========================
        # 4. Feature Engineering for MLP (Stream B)
        # ==========================
        print("Preparing MLP features...")

        # Scale Metadata (Arcsinh + StandardScaler)
        # Arcsinh handles skewed distributions (karma, age) better than log (handles 0 and neg)
        transformer = FunctionTransformer(np.arcsinh)
        meta_train_trans = transformer.transform(meta_train_imp)  # Use imputed version
        meta_val_trans = transformer.transform(meta_val_imp)
        meta_test_trans = transformer.transform(meta_test_imp)

        self.scaler.fit(meta_train_trans)
        meta_train_sc = self.scaler.transform(meta_train_trans)
        meta_val_sc = self.scaler.transform(meta_val_trans)
        meta_test_sc = self.scaler.transform(meta_test_trans)

        # Concatenate Scaled Meta + Binary TopK
        dense_train = np.hstack([meta_train_sc, topk_train])
        dense_val = np.hstack([meta_val_sc, topk_val])
        dense_test = np.hstack([meta_test_sc, topk_test])

        # Pad History Sequences
        pad_len = 50
        seq_train = self._pad_sequences(train_hist_seqs, max_len=pad_len)
        seq_val = self._pad_sequences(val_hist_seqs, max_len=pad_len)
        seq_test = self._pad_sequences(test_hist_seqs, max_len=pad_len)

        # ==========================
        # 5. Saving & Return
        # ==========================
        rf_data = {
            "X_train": X_train_rf.astype(np.float32),
            "y_train": y_train,
            "X_val": X_val_rf.astype(np.float32),
            "y_val": y_val,
            "X_test": X_test_rf.astype(np.float32),
            "test_ids": df_test["request_id"].values,
        }

        mlp_data = {
            "train_title_emb": train_title_emb.astype(np.float32),
            "train_body_emb": train_body_emb.astype(np.float32),
            "train_hist_seq": seq_train,
            "train_dense": dense_train.astype(np.float32),
            "y_train": y_train,
            "val_title_emb": val_title_emb.astype(np.float32),
            "val_body_emb": val_body_emb.astype(np.float32),
            "val_hist_seq": seq_val,
            "val_dense": dense_val.astype(np.float32),
            "y_val": y_val,
            "test_title_emb": test_title_emb.astype(np.float32),
            "test_body_emb": test_body_emb.astype(np.float32),
            "test_hist_seq": seq_test,
            "test_dense": dense_test.astype(np.float32),
            "test_ids": df_test["request_id"].values,
        }

        print("Saving features to cache...")
        np.savez(self.rf_features_path, **rf_data)
        np.savez(self.mlp_features_path, **mlp_data)

        return rf_data, mlp_data
