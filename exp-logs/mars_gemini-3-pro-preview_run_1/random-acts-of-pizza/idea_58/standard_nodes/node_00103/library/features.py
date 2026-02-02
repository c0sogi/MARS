import os
import ast
import numpy as np
import pandas as pd
import torch
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import load_dataset

# Ensure VADER lexicon is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class FeatureGenerator:
    def __init__(self):
        self.device = Config.DEVICE
        self.sbert = SentenceTransformer(Config.SBERT_MODEL_NAME, device=self.device)
        self.tfidf = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        self.scaler = StandardScaler()
        self.vader = SentimentIntensityAnalyzer()

        # State
        self.top_k_subreddits = []
        self.numeric_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]
        self.subreddit_embedding_cache = {}

    def fit(self, df_train):
        """Fits scalers, TF-IDF, and identifies Top-K subreddits using training data."""
        print("Fitting FeatureGenerator on training data...")

        # 1. Fit TF-IDF
        text_corpus = (
            df_train[Config.TEXT_TITLE_COL].fillna("")
            + " "
            + df_train[Config.TEXT_BODY_COL].fillna("")
        )
        self.tfidf.fit(text_corpus)

        # 2. Fit Scaler (on arcsinh transformed data)
        meta_data = self._preprocess_metadata(df_train)
        self.scaler.fit(meta_data)

        # 3. Identify Top-K Subreddits
        all_subreddits = []
        for sub_list_str in df_train["requester_subreddits_at_request"]:
            try:
                subs = (
                    ast.literal_eval(sub_list_str)
                    if isinstance(sub_list_str, str)
                    else []
                )
                all_subreddits.extend(subs)
            except:
                continue

        counts = pd.Series(all_subreddits).value_counts()
        self.top_k_subreddits = counts.head(Config.TOP_K_SUBREDDITS).index.tolist()

        print(
            f"FeatureGenerator fitted. Top-K Subreddits: {len(self.top_k_subreddits)}"
        )

    def transform(self, df):
        """Transforms a dataset into features for MLP and RF."""
        print(f"Transforming dataset with {len(df)} samples...")

        # --- 1. Text Embeddings (SBERT) ---
        titles = df[Config.TEXT_TITLE_COL].fillna("").tolist()
        bodies = df[Config.TEXT_BODY_COL].fillna("").tolist()

        title_emb = self.sbert.encode(
            titles, convert_to_numpy=True, show_progress_bar=False
        )
        body_emb = self.sbert.encode(
            bodies, convert_to_numpy=True, show_progress_bar=False
        )

        # --- 2. TF-IDF (RF) ---
        text_combined = [t + " " + b for t, b in zip(titles, bodies)]
        tfidf_feats = self.tfidf.transform(text_combined).toarray().astype(np.float32)

        # --- 3. Metadata Processing ---
        # Raw metadata for interaction features
        meta_raw = df[self.numeric_cols].fillna(0).copy()
        # Scaled metadata for MLP
        meta_scaled = self.scaler.transform(self._preprocess_metadata(df)).astype(
            np.float32
        )

        # --- 4. Top-K Indicators ---
        top_k_feats = np.zeros((len(df), len(self.top_k_subreddits)), dtype=np.float32)

        # --- 5. History & Consistency ---
        history_centroids = []
        consistency_titles = []
        consistency_bodies = []

        # Pre-compute embeddings for all unique subreddits in this batch to save time
        batch_subreddits = []
        parsed_subs_list = []
        for sub_list_str in df["requester_subreddits_at_request"]:
            try:
                subs = (
                    ast.literal_eval(sub_list_str)
                    if isinstance(sub_list_str, str)
                    else []
                )
            except:
                subs = []
            parsed_subs_list.append(subs)
            batch_subreddits.extend(subs)

        unique_subs = list(set(batch_subreddits))
        # Filter out those already cached
        new_subs = [s for s in unique_subs if s not in self.subreddit_embedding_cache]
        if new_subs:
            new_embs = self.sbert.encode(
                new_subs, convert_to_numpy=True, show_progress_bar=False
            )
            for s, e in zip(new_subs, new_embs):
                self.subreddit_embedding_cache[s] = e

        # Compute features per row
        for i, subs in enumerate(parsed_subs_list):
            # Top-K
            for j, top_sub in enumerate(self.top_k_subreddits):
                if top_sub in subs:
                    top_k_feats[i, j] = 1.0

            # History Centroid
            if subs:
                sub_embs = np.array(
                    [
                        self.subreddit_embedding_cache[s]
                        for s in subs
                        if s in self.subreddit_embedding_cache
                    ]
                )
                if len(sub_embs) > 0:
                    centroid = np.mean(sub_embs, axis=0)
                else:
                    centroid = np.zeros(Config.SBERT_EMBEDDING_DIM)
            else:
                centroid = np.zeros(Config.SBERT_EMBEDDING_DIM)
            history_centroids.append(centroid)

            # Consistency (Cosine Sim)
            # Avoid divide by zero
            norm_centroid = np.linalg.norm(centroid)
            norm_title = np.linalg.norm(title_emb[i])
            norm_body = np.linalg.norm(body_emb[i])

            sim_t = (
                np.dot(centroid, title_emb[i]) / (norm_centroid * norm_title + 1e-9)
                if norm_centroid > 0
                else 0.0
            )
            sim_b = (
                np.dot(centroid, body_emb[i]) / (norm_centroid * norm_body + 1e-9)
                if norm_centroid > 0
                else 0.0
            )

            consistency_titles.append(sim_t)
            consistency_bodies.append(sim_b)

        history_centroids = np.array(history_centroids, dtype=np.float32)
        consistency_feats = np.column_stack(
            [consistency_titles, consistency_bodies]
        ).astype(np.float32)

        # --- 6. Sentiment Analysis ---
        sentiments = []
        for t, b in zip(titles, bodies):
            s_t = self.vader.polarity_scores(t)
            s_b = self.vader.polarity_scores(b)
            # Vector: [Title_Compound, Title_Neg, Body_Compound, Body_Neg]
            sentiments.append(
                [s_t["compound"], s_t["neg"], s_b["compound"], s_b["neg"]]
            )
        sentiment_feats = np.array(sentiments, dtype=np.float32)

        # --- 7. Multi-Axis Interaction Block (RF Only) ---
        # Derived Metrics
        # Upvote Ratio: (Up / (Up + Down)). Approx: Up+Down = Total, Up-Down = Diff => 2*Up = Total+Diff
        total_votes = meta_raw["requester_upvotes_plus_downvotes_at_request"]
        diff_votes = meta_raw["requester_upvotes_minus_downvotes_at_request"]
        # Avoid zero division
        upvotes = (total_votes + diff_votes) / 2
        upvote_ratio = (upvotes / (total_votes + 1e-9)).clip(0, 1)

        account_age = meta_raw["requester_account_age_in_days_at_request"]
        log_age = np.log1p(account_age)

        text_len = np.array([len(b) for b in bodies])

        # Interactions
        # 1. Consistency * Credibility (Age)
        int_cons_age = consistency_feats[:, 0] * log_age
        # 2. Consistency * Credibility (Ratio)
        int_cons_ratio = consistency_feats[:, 1] * upvote_ratio
        # 3. Sentiment (Neg) * Credibility (Ratio)
        int_sent_ratio = sentiment_feats[:, 3] * upvote_ratio
        # 4. Structure (Len) * Credibility (Age)
        int_struct_age = np.log1p(text_len) * log_age

        interaction_feats = np.column_stack(
            [int_cons_age, int_cons_ratio, int_sent_ratio, int_struct_age]
        ).astype(np.float32)

        # --- 8. Assemble Outputs ---

        # For MLP: Dictionary of tensors
        mlp_features = {
            "title_emb": title_emb,
            "body_emb": body_emb,
            "metadata": meta_scaled,
            "top_k": top_k_feats,
            "history_centroid": history_centroids,
            "consistency": consistency_feats,
            "sentiment": sentiment_feats,
        }

        # For RF: Concatenated Array
        # [TFIDF (5000), Metadata (Scaled), Top-K (50), Interactions (4), Sentiment (4), Consistency (2)]
        # Note: We use scaled metadata for RF too as it helps convergence, though RF is robust.
        rf_features = np.hstack(
            [
                tfidf_feats,
                meta_scaled,
                top_k_feats,
                interaction_feats,
                sentiment_feats,
                consistency_feats,
            ]
        ).astype(np.float32)

        # Labels
        if Config.TARGET_COL in df.columns:
            labels = df[Config.TARGET_COL].astype(int).values
        else:
            labels = np.zeros(len(df))  # Dummy for test

        return mlp_features, rf_features, labels

    def _preprocess_metadata(self, df):
        """Applies arcsinh transform to numeric columns."""
        data = df[self.numeric_cols].fillna(0).copy()
        # Apply arcsinh to handle skewness in counts/days
        return np.arcsinh(data)


def get_features(load_cached_data=True):
    """
    Main entry point. Handles loading data, running the pipeline, and caching.
    Returns:
        (train_data, val_data, test_data)
        Each is a tuple: (mlp_dict, rf_array, labels)
    """
    # Define cache paths
    cache_dir = Config.IDEA_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_train = os.path.join(cache_dir, "features_train.npz")
    path_val = os.path.join(cache_dir, "features_val.npz")
    path_test = os.path.join(cache_dir, "features_test.npz")

    # Check if all caches exist
    all_cached = (
        os.path.exists(path_train)
        and os.path.exists(path_val)
        and os.path.exists(path_test)
    )

    if load_cached_data and all_cached:
        print("Loading features from cache...")
        try:
            # Helper to load npz back to dict/arrays
            def load_split(path):
                data = np.load(path, allow_pickle=True)
                # Reconstruct MLP dict
                mlp_dict = {
                    k: data[f"mlp_{k}"]
                    for k in [
                        "title_emb",
                        "body_emb",
                        "metadata",
                        "top_k",
                        "history_centroid",
                        "consistency",
                        "sentiment",
                    ]
                }
                rf_arr = data["rf_features"]
                labels = data["labels"]
                return mlp_dict, rf_arr, labels

            return load_split(path_train), load_split(path_val), load_split(path_test)
        except Exception as e:
            print(f"Cache load failed ({e}). Recomputing...")

    # Compute from scratch
    print("Computing features from scratch...")

    # Load raw data
    df_train = load_dataset("train")
    df_val = load_dataset("val")
    df_test = load_dataset("test")

    # Initialize and Fit Pipeline
    pipeline = FeatureGenerator()
    pipeline.fit(df_train)

    # Transform
    train_mlp, train_rf, train_y = pipeline.transform(df_train)
    val_mlp, val_rf, val_y = pipeline.transform(df_val)
    test_mlp, test_rf, test_y = pipeline.transform(df_test)

    # Save to Cache
    def save_split(path, mlp_dict, rf_arr, labels):
        save_dict = {f"mlp_{k}": v for k, v in mlp_dict.items()}
        save_dict["rf_features"] = rf_arr
        save_dict["labels"] = labels
        np.savez(path, **save_dict)

    save_split(path_train, train_mlp, train_rf, train_y)
    save_split(path_val, val_mlp, val_rf, val_y)
    save_split(path_test, test_mlp, test_rf, test_y)

    print(f"Features computed and saved to {cache_dir}")

    return (
        (train_mlp, train_rf, train_y),
        (val_mlp, val_rf, val_y),
        (test_mlp, test_rf, test_y),
    )
