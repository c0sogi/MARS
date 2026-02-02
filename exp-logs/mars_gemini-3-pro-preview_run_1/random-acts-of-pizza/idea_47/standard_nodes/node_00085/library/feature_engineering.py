import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

# Import provided library
from library.config import (
    WORKING_DIR,
    SBERT_MODEL_NAME,
    TOP_K_SUBREDDITS,
    DEVICE,
    TARGET_COL,
)
from library.utils import seed_everything
from library.data_loader import load_dataset

# Ensure NLTK resources
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    try:
        nltk.download("vader_lexicon", quiet=True)
    except:
        pass


class TextProcessor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        try:
            self.vader = SentimentIntensityAnalyzer()
        except:
            self.vader = None

    def get_tfidf_features(self, train_text, val_text, test_text):
        print("Generating TF-IDF features...")
        train_tfidf = self.tfidf.fit_transform(train_text)
        val_tfidf = self.tfidf.transform(val_text)
        test_tfidf = self.tfidf.transform(test_text)
        return train_tfidf, val_tfidf, test_tfidf

    def get_sentiment_features(self, df):
        print("Generating Sentiment features...")
        if self.vader is None:
            return np.zeros((len(df), 4))

        texts = (
            df["request_title"].fillna("")
            + " "
            + df["request_text_edit_aware"].fillna("")
        ).astype(str)

        sentiments = []
        for text in texts:
            scores = self.vader.polarity_scores(text)
            sentiments.append(
                [scores["neg"], scores["neu"], scores["pos"], scores["compound"]]
            )
        return np.array(sentiments)


class SBERTExtractor:
    def __init__(self):
        self.model = SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE)
        self.history_max_len = 50

    def encode(self, texts, batch_size=32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def process_history(self, subreddits_series):
        print("Processing User History Embeddings...")
        all_subs = set()
        for subs in subreddits_series:
            all_subs.update(subs)

        unique_subs = list(all_subs)
        if not unique_subs:
            sub_to_emb = {}
        else:
            unique_subs_text = [f"subreddit {s}" for s in unique_subs]
            sub_embs = self.encode(unique_subs_text, batch_size=128)
            sub_to_emb = {s: emb for s, emb in zip(unique_subs, sub_embs)}

        N = len(subreddits_series)
        dim = self.model.get_sentence_embedding_dimension()

        history_emb = np.zeros((N, self.history_max_len, dim), dtype=np.float32)
        history_mask = np.zeros((N, self.history_max_len), dtype=np.float32)
        centroid = np.zeros((N, dim), dtype=np.float32)

        for i, subs in enumerate(subreddits_series):
            if len(subs) == 0:
                continue

            current_subs = subs[: self.history_max_len]
            user_embs = []
            for j, s in enumerate(current_subs):
                if s in sub_to_emb:
                    emb = sub_to_emb[s]
                    history_emb[i, j, :] = emb
                    history_mask[i, j] = 1.0
                    user_embs.append(emb)

            if user_embs:
                centroid[i, :] = np.mean(user_embs, axis=0)

        return history_emb, history_mask, centroid


class TopKProfiler:
    def __init__(self, k=TOP_K_SUBREDDITS):
        self.k = k
        self.top_subs = []

    def fit(self, train_subreddits):
        counter = Counter()
        for subs in train_subreddits:
            counter.update(subs)
        self.top_subs = [s for s, _ in counter.most_common(self.k)]

    def transform(self, subreddits_series):
        matrix = np.zeros((len(subreddits_series), self.k), dtype=int)
        for i, subs in enumerate(subreddits_series):
            s_set = set(subs)
            for j, top_s in enumerate(self.top_subs):
                if top_s in s_set:
                    matrix[i, j] = 1
        return matrix


class MetadataProcessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.numeric_cols = [
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

    def get_numeric_features(self, df, fit=False):
        data = df[self.numeric_cols].copy()
        if fit:
            data = self.imputer.fit_transform(data)
        else:
            data = self.imputer.transform(data)
        return data

    def get_interaction_features(self, numeric_data, title_emb, body_emb, centroid_emb):
        print("Generating Interaction Features...")
        title_consistency = np.sum(title_emb * centroid_emb, axis=1)
        body_consistency = np.sum(body_emb * centroid_emb, axis=1)

        interactions = []
        for col_idx in range(numeric_data.shape[1]):
            col_vals = numeric_data[:, col_idx]
            min_val = np.min(col_vals)
            offset = 0
            if min_val < 0:
                offset = abs(min_val)

            log_vals = np.log1p(col_vals + offset)
            interactions.append(title_consistency * log_vals)
            interactions.append(body_consistency * log_vals)

        return np.stack(interactions, axis=1)


class FeaturePipeline:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.sbert = SBERTExtractor()
        self.topk = TopKProfiler()
        self.meta_processor = MetadataProcessor()

    def run(self, load_cached_data=True):
        seed_everything()
        os.makedirs(WORKING_DIR, exist_ok=True)
        rf_cache_path = os.path.join(WORKING_DIR, "rf_features.npz")
        mlp_cache_path = os.path.join(WORKING_DIR, "mlp_features.npz")

        if (
            load_cached_data
            and os.path.exists(rf_cache_path)
            and os.path.exists(mlp_cache_path)
        ):
            print("Loading features from cache...")
            rf_data = np.load(rf_cache_path, allow_pickle=True)
            mlp_data = np.load(mlp_cache_path, allow_pickle=True)

            rf_out = {k: rf_data[k] for k in rf_data.files}
            mlp_out = {k: mlp_data[k] for k in mlp_data.files}
            return rf_out, mlp_out

        print("Computing features from scratch...")
        train_df, val_df, test_df = load_dataset(load_cached_data=True)

        # Text Processing
        get_text = lambda df: (
            df["request_title"].fillna("")
            + " "
            + df["request_text_edit_aware"].fillna("")
        ).tolist()
        train_tfidf, val_tfidf, test_tfidf = self.text_processor.get_tfidf_features(
            get_text(train_df), get_text(val_df), get_text(test_df)
        )

        train_sent = self.text_processor.get_sentiment_features(train_df)
        val_sent = self.text_processor.get_sentiment_features(val_df)
        test_sent = self.text_processor.get_sentiment_features(test_df)

        # SBERT Embeddings
        print("Encoding Titles...")
        train_title_emb = self.sbert.encode(
            train_df["request_title"].fillna("").tolist()
        )
        val_title_emb = self.sbert.encode(val_df["request_title"].fillna("").tolist())
        test_title_emb = self.sbert.encode(test_df["request_title"].fillna("").tolist())

        print("Encoding Bodies...")
        train_body_emb = self.sbert.encode(
            train_df["request_text_edit_aware"].fillna("").tolist()
        )
        val_body_emb = self.sbert.encode(
            val_df["request_text_edit_aware"].fillna("").tolist()
        )
        test_body_emb = self.sbert.encode(
            test_df["request_text_edit_aware"].fillna("").tolist()
        )

        print("Processing History...")
        train_hist_emb, train_hist_mask, train_centroid = self.sbert.process_history(
            train_df["requester_subreddits_at_request"]
        )
        val_hist_emb, val_hist_mask, val_centroid = self.sbert.process_history(
            val_df["requester_subreddits_at_request"]
        )
        test_hist_emb, test_hist_mask, test_centroid = self.sbert.process_history(
            test_df["requester_subreddits_at_request"]
        )

        # Top-K Subreddits
        print("Generating Top-K Subreddit features...")
        self.topk.fit(train_df["requester_subreddits_at_request"])
        train_topk = self.topk.transform(train_df["requester_subreddits_at_request"])
        val_topk = self.topk.transform(val_df["requester_subreddits_at_request"])
        test_topk = self.topk.transform(test_df["requester_subreddits_at_request"])

        # Metadata & Interactions
        print("Processing Metadata...")
        train_num = self.meta_processor.get_numeric_features(train_df, fit=True)
        val_num = self.meta_processor.get_numeric_features(val_df, fit=False)
        test_num = self.meta_processor.get_numeric_features(test_df, fit=False)

        train_inter = self.meta_processor.get_interaction_features(
            train_num, train_title_emb, train_body_emb, train_centroid
        )
        val_inter = self.meta_processor.get_interaction_features(
            val_num, val_title_emb, val_body_emb, val_centroid
        )
        test_inter = self.meta_processor.get_interaction_features(
            test_num, test_title_emb, test_body_emb, test_centroid
        )

        # Scaled Numeric for MLP (Arcsinh + StandardScale)
        train_num_arcsinh = np.arcsinh(train_num)
        val_num_arcsinh = np.arcsinh(val_num)
        test_num_arcsinh = np.arcsinh(test_num)

        scaler = StandardScaler()
        train_num_scaled = scaler.fit_transform(train_num_arcsinh)
        val_num_scaled = scaler.transform(val_num_arcsinh)
        test_num_scaled = scaler.transform(test_num_arcsinh)

        # Assemble Datasets
        def assemble_rf(tfidf, num, topk, sent, inter):
            return np.hstack([tfidf.toarray(), num, topk, sent, inter]).astype(
                np.float32
            )

        rf_train_X = assemble_rf(
            train_tfidf, train_num, train_topk, train_sent, train_inter
        )
        rf_val_X = assemble_rf(val_tfidf, val_num, val_topk, val_sent, val_inter)
        rf_test_X = assemble_rf(test_tfidf, test_num, test_topk, test_sent, test_inter)

        rf_out = {
            "train_X": rf_train_X,
            "train_y": train_df[TARGET_COL].values,
            "val_X": rf_val_X,
            "val_y": val_df[TARGET_COL].values,
            "test_X": rf_test_X,
        }

        mlp_out = {
            "train_title": train_title_emb,
            "train_body": train_body_emb,
            "train_hist": train_hist_emb,
            "train_hist_mask": train_hist_mask,
            "train_centroid": train_centroid,
            "train_meta": train_num_scaled,
            "train_y": train_df[TARGET_COL].values,
            "val_title": val_title_emb,
            "val_body": val_body_emb,
            "val_hist": val_hist_emb,
            "val_hist_mask": val_hist_mask,
            "val_centroid": val_centroid,
            "val_meta": val_num_scaled,
            "val_y": val_df[TARGET_COL].values,
            "test_title": test_title_emb,
            "test_body": test_body_emb,
            "test_hist": test_hist_emb,
            "test_hist_mask": test_hist_mask,
            "test_centroid": test_centroid,
            "test_meta": test_num_scaled,
        }

        print("Saving features to cache...")
        np.savez_compressed(rf_cache_path, **rf_out)
        np.savez_compressed(mlp_cache_path, **mlp_out)

        return rf_out, mlp_out
