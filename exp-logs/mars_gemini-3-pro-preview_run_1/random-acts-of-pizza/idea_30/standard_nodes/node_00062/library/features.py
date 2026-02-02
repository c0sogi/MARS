import os
import json
import numpy as np
import pandas as pd
import torch
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

from library.config import Config
from library.utils import ensure_dir


class FeatureEngineer:
    """
    Handles feature engineering for both Stream A (Random Forest) and Stream B (MLP).
    Implements caching, text processing, and metadata extraction.
    """

    def __init__(self):
        # Initialize models and tools
        print("Initializing FeatureEngineer components...")
        self.sbert = SentenceTransformer(Config.SBERT_MODEL_NAME)

        # Download VADER lexicon if needed
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        self.vader = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(
            max_features=Config.TFIDF_VOCAB_SIZE,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
            dtype=np.float32,
        )
        self.top_k_subreddits = []

    def load_raw_data(self):
        """Loads raw CSVs from metadata directory."""
        print("Loading raw metadata CSVs...")
        train = pd.read_csv(Config.TRAIN_PATH)
        val = pd.read_csv(Config.VAL_PATH)
        test = pd.read_csv(Config.TEST_PATH)
        return train, val, test

    def _parse_subreddits(self, df):
        """Parses the string representation of subreddit lists."""

        def parse(x):
            try:
                if isinstance(x, str):
                    return eval(x)
                return x if isinstance(x, list) else []
            except:
                return []

        return df["requester_subreddits_at_request"].apply(parse)

    def _get_text_features(self, df):
        """Combines title and body for text processing."""
        return (
            df["request_title"].fillna("")
            + " "
            + df["request_text_edit_aware"].fillna("")
        ).tolist()

    def _extract_rf_metadata(self, df):
        """Extracts numerical and sentiment features for Random Forest."""
        out = pd.DataFrame()

        # 1. Raw Numerical Features
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

        for c in num_cols:
            # Fill NaNs with 0 for counts/days
            out[c] = df[c].fillna(0)

        # 2. Engineered Ratios
        # Add epsilon to avoid division by zero
        eps = 1e-5
        out["upvote_ratio"] = df["requester_upvotes_minus_downvotes_at_request"] / (
            df["requester_upvotes_plus_downvotes_at_request"] + eps
        )
        out["raop_comment_ratio"] = df[
            "requester_number_of_comments_in_raop_at_request"
        ] / (df["requester_number_of_comments_at_request"] + eps)
        out["raop_post_ratio"] = df["requester_number_of_posts_on_raop_at_request"] / (
            df["requester_number_of_posts_at_request"] + eps
        )

        # 3. Text Meta-Features
        texts = df["request_text_edit_aware"].fillna("").astype(str)
        out["text_len_char"] = texts.apply(len)
        out["text_len_word"] = texts.apply(lambda x: len(x.split()))
        out["text_caps_ratio"] = texts.apply(
            lambda x: sum(1 for c in x if c.isupper()) / (len(x) + eps)
        )

        # 4. Sentiment Analysis (VADER)
        # VADER is fast enough for this dataset size
        sentiments = texts.apply(lambda x: self.vader.polarity_scores(x))
        out["sent_neg"] = sentiments.apply(lambda x: x["neg"])
        out["sent_neu"] = sentiments.apply(lambda x: x["neu"])
        out["sent_pos"] = sentiments.apply(lambda x: x["pos"])
        out["sent_compound"] = sentiments.apply(lambda x: x["compound"])

        return out

    def _compute_alignment_scalar(self, df, subs_series):
        """
        Computes the Cosine Similarity between the Request Embedding and the
        Centroid of the User's History Embeddings.
        """
        # Embed Requests
        req_texts = self._get_text_features(df)
        req_embs = self.sbert.encode(
            req_texts, convert_to_numpy=True, show_progress_bar=False
        )

        # Identify unique subreddits in this batch to minimize encoding
        batch_subs = set([s for sublist in subs_series for s in sublist])
        sub_map = {}
        if batch_subs:
            unique_subs = list(batch_subs)
            unique_embs = self.sbert.encode(
                unique_subs, convert_to_numpy=True, show_progress_bar=False
            )
            sub_map = {s: e for s, e in zip(unique_subs, unique_embs)}

        # Compute Centroids
        centroids = np.zeros_like(req_embs)
        for i, subs in enumerate(subs_series):
            if not subs:
                continue
            # Average embeddings of history
            valid_embs = [sub_map[s] for s in subs if s in sub_map]
            if valid_embs:
                centroids[i] = np.mean(valid_embs, axis=0)

        # Cosine Similarity
        # Normalize
        req_norm = np.linalg.norm(req_embs, axis=1, keepdims=True) + 1e-9
        cent_norm = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9

        req_unit = req_embs / req_norm
        cent_unit = centroids / cent_norm

        # Dot product
        scores = np.sum(req_unit * cent_unit, axis=1)

        return pd.DataFrame(scores, columns=["alignment_score"])

    def process_stream_a(self, load_cached_data=True):
        """
        Generates features for Stream A (Random Forest).
        Returns: (train_df, val_df, test_df)
        """
        if load_cached_data and os.path.exists(Config.CACHE_RF_TRAIN):
            print("Loading cached Stream A (RF) data...")
            return (
                pd.read_parquet(Config.CACHE_RF_TRAIN),
                pd.read_parquet(Config.CACHE_RF_VAL),
                pd.read_parquet(Config.CACHE_RF_TEST),
            )

        print("Generating Stream A (RF) features...")
        train, val, test = self.load_raw_data()

        # 1. Metadata
        train_meta = self._extract_rf_metadata(train)
        val_meta = self._extract_rf_metadata(val)
        test_meta = self._extract_rf_metadata(test)

        # 2. Top-K Subreddits
        train_subs = self._parse_subreddits(train)
        val_subs = self._parse_subreddits(val)
        test_subs = self._parse_subreddits(test)

        # Determine Top-K from Train
        all_subs = [s for sublist in train_subs for s in sublist]
        top_k = [s for s, c in Counter(all_subs).most_common(Config.TOP_K_SUBREDDITS)]

        def get_top_k_feats(subs_series):
            matrix = np.zeros((len(subs_series), len(top_k)), dtype=int)
            for i, user_subs in enumerate(subs_series):
                s_set = set(user_subs)
                for j, k_sub in enumerate(top_k):
                    if k_sub in s_set:
                        matrix[i, j] = 1
            return pd.DataFrame(matrix, columns=[f"sub_{s}" for s in top_k])

        train_topk = get_top_k_feats(train_subs)
        val_topk = get_top_k_feats(val_subs)
        test_topk = get_top_k_feats(test_subs)

        # 3. TF-IDF
        train_text = self._get_text_features(train)
        val_text = self._get_text_features(val)
        test_text = self._get_text_features(test)

        print("Fitting TF-IDF...")
        self.tfidf.fit(train_text)

        # Transform and convert to DataFrame (Dense)
        # Using float32 to save space
        train_tfidf = pd.DataFrame(
            self.tfidf.transform(train_text).toarray(),
            columns=self.tfidf.get_feature_names_out(),
        )
        val_tfidf = pd.DataFrame(
            self.tfidf.transform(val_text).toarray(),
            columns=self.tfidf.get_feature_names_out(),
        )
        test_tfidf = pd.DataFrame(
            self.tfidf.transform(test_text).toarray(),
            columns=self.tfidf.get_feature_names_out(),
        )

        # 4. Alignment Scalar
        print("Computing Alignment Scalars for RF...")
        train_align = self._compute_alignment_scalar(train, train_subs)
        val_align = self._compute_alignment_scalar(val, val_subs)
        test_align = self._compute_alignment_scalar(test, test_subs)

        # Concatenate
        # Ensure indices match
        for df_list in [
            [train_meta, train_topk, train_tfidf, train_align],
            [val_meta, val_topk, val_tfidf, val_align],
            [test_meta, test_topk, test_tfidf, test_align],
        ]:
            for d in df_list:
                d.reset_index(drop=True, inplace=True)

        X_train = pd.concat([train_meta, train_topk, train_tfidf, train_align], axis=1)
        X_val = pd.concat([val_meta, val_topk, val_tfidf, val_align], axis=1)
        X_test = pd.concat([test_meta, test_topk, test_tfidf, test_align], axis=1)

        # Attach Targets
        X_train["requester_received_pizza"] = train["requester_received_pizza"].astype(
            int
        )
        X_val["requester_received_pizza"] = val["requester_received_pizza"].astype(int)

        # Cache
        print("Caching Stream A data...")
        ensure_dir(Config.CACHE_RF_TRAIN)
        X_train.to_parquet(Config.CACHE_RF_TRAIN)
        X_val.to_parquet(Config.CACHE_RF_VAL)
        X_test.to_parquet(Config.CACHE_RF_TEST)

        return X_train, X_val, X_test

    def process_stream_b(self, load_cached_data=True):
        """
        Generates features for Stream B (MLP).
        Returns: (train_dict, val_dict, test_dict)
        """
        if load_cached_data and os.path.exists(Config.CACHE_MLP_TRAIN):
            print("Loading cached Stream B (MLP) data...")
            return (
                np.load(Config.CACHE_MLP_TRAIN),
                np.load(Config.CACHE_MLP_VAL),
                np.load(Config.CACHE_MLP_TEST),
            )

        print("Generating Stream B (MLP) features...")
        train, val, test = self.load_raw_data()

        # 1. Metadata (Arcsinh + Scaled)
        # Use same extraction logic as RF but we will transform it
        train_meta = self._extract_rf_metadata(train)
        val_meta = self._extract_rf_metadata(val)
        test_meta = self._extract_rf_metadata(test)

        # Arcsinh transform
        train_meta = np.arcsinh(train_meta)
        val_meta = np.arcsinh(val_meta)
        test_meta = np.arcsinh(test_meta)

        # Scale
        self.scaler.fit(train_meta)
        train_meta_scaled = self.scaler.transform(train_meta).astype(np.float32)
        val_meta_scaled = self.scaler.transform(val_meta).astype(np.float32)
        test_meta_scaled = self.scaler.transform(test_meta).astype(np.float32)

        # 2. Text Embeddings (Title, Body)
        print("Encoding Text Embeddings...")
        train_title = self.sbert.encode(
            train["request_title"].fillna("").tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        val_title = self.sbert.encode(
            val["request_title"].fillna("").tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        test_title = self.sbert.encode(
            test["request_title"].fillna("").tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        train_body = self.sbert.encode(
            train["request_text_edit_aware"].fillna("").tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        val_body = self.sbert.encode(
            val["request_text_edit_aware"].fillna("").tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        test_body = self.sbert.encode(
            test["request_text_edit_aware"].fillna("").tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # 3. History Sequence Embeddings
        print("Encoding History Sequences...")
        train_subs = self._parse_subreddits(train)
        val_subs = self._parse_subreddits(val)
        test_subs = self._parse_subreddits(test)

        # Map all unique subreddits
        all_subs = set(
            [
                s
                for sublist in pd.concat([train_subs, val_subs, test_subs])
                for s in sublist
            ]
        )
        sub_map = {}
        if all_subs:
            unique_subs = list(all_subs)
            unique_embs = self.sbert.encode(
                unique_subs, convert_to_numpy=True, show_progress_bar=False
            )
            sub_map = {s: e for s, e in zip(unique_subs, unique_embs)}

        # Create Tensors
        MAX_SEQ_LEN = 50
        EMB_DIM = Config.MLP_EMBEDDING_DIM

        def create_seq_tensor(subs_series):
            N = len(subs_series)
            tensor = np.zeros((N, MAX_SEQ_LEN, EMB_DIM), dtype=np.float32)
            mask = np.zeros((N, MAX_SEQ_LEN), dtype=np.float32)

            for i, subs in enumerate(subs_series):
                # Take up to MAX_SEQ_LEN
                curr = subs[:MAX_SEQ_LEN]
                for j, s in enumerate(curr):
                    if s in sub_map:
                        tensor[i, j] = sub_map[s]
                        mask[i, j] = 1.0
            return tensor, mask

        train_hist, train_mask = create_seq_tensor(train_subs)
        val_hist, val_mask = create_seq_tensor(val_subs)
        test_hist, test_mask = create_seq_tensor(test_subs)

        # Targets
        y_train = train["requester_received_pizza"].astype(int).values
        y_val = val["requester_received_pizza"].astype(int).values

        # Pack
        train_data = {
            "meta": train_meta_scaled,
            "title_emb": train_title,
            "body_emb": train_body,
            "hist_emb": train_hist,
            "hist_mask": train_mask,
            "y": y_train,
        }
        val_data = {
            "meta": val_meta_scaled,
            "title_emb": val_title,
            "body_emb": val_body,
            "hist_emb": val_hist,
            "hist_mask": val_mask,
            "y": y_val,
        }
        test_data = {
            "meta": test_meta_scaled,
            "title_emb": test_title,
            "body_emb": test_body,
            "hist_emb": test_hist,
            "hist_mask": test_mask,
        }

        # Cache
        print("Caching Stream B data...")
        ensure_dir(Config.CACHE_MLP_TRAIN)
        np.savez(Config.CACHE_MLP_TRAIN, **train_data)
        np.savez(Config.CACHE_MLP_VAL, **val_data)
        np.savez(Config.CACHE_MLP_TEST, **test_data)

        return train_data, val_data, test_data
