import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from scipy import sparse
import torch
from library.config import Config
from library.data_utils import load_data


class MetadataProcessor:
    """
    Handles processing of numerical metadata for both Random Forest (Stream A)
    and MLP (Stream B).
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.numeric_cols = Config.NUMERIC_COLS
        self.is_fitted = False

    def _generate_derived_features(self, df):
        """
        Generates derived ratio features from raw metadata.
        """
        # Ensure we are working with a copy to avoid SettingWithCopy warnings
        df_eng = df[self.numeric_cols].copy()

        # Upvote Ratio
        # plus = up + down, minus = up - down
        # up = (plus + minus) / 2
        # ratio = up / (plus + epsilon)
        plus = df_eng["requester_upvotes_plus_downvotes_at_request"]
        minus = df_eng["requester_upvotes_minus_downvotes_at_request"]
        upvotes = (plus + minus) / 2.0

        # Avoid division by zero
        epsilon = 1e-6
        df_eng["upvote_ratio"] = upvotes / (plus + epsilon)

        # Interaction Ratios
        # Comments per post
        n_posts = df_eng["requester_number_of_posts_at_request"]
        n_comments = df_eng["requester_number_of_comments_at_request"]
        df_eng["comment_to_post_ratio"] = n_comments / (n_posts + 1.0)

        # RAOP Activity Ratio
        raop_comments = df_eng["requester_number_of_comments_in_raop_at_request"]
        df_eng["raop_activity_ratio"] = raop_comments / (n_comments + 1.0)

        return df_eng

    def fit(self, train_df):
        """
        Fits imputer and scaler on training data.
        """
        train_eng = self._generate_derived_features(train_df)
        self.imputer.fit(train_eng)

        # For MLP, we fit scaler on imputed, arcsinh-transformed data
        train_imputed = self.imputer.transform(train_eng)
        train_arcsinh = np.arcsinh(train_imputed)
        self.scaler.fit(train_arcsinh)

        self.is_fitted = True

    def transform_rf(self, df):
        """
        Transforms data for Random Forest (Imputation + Derived Features).
        Returns numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("MetadataProcessor must be fitted before transform.")

        df_eng = self._generate_derived_features(df)
        # RF handles raw magnitudes well, but needs no NaNs
        return self.imputer.transform(df_eng)

    def transform_mlp(self, df):
        """
        Transforms data for MLP (Imputation + Arcsinh + Scaling).
        Returns numpy array.
        """
        if not self.is_fitted:
            raise RuntimeError("MetadataProcessor must be fitted before transform.")

        df_eng = self._generate_derived_features(df)
        data_imputed = self.imputer.transform(df_eng)
        data_arcsinh = np.arcsinh(data_imputed)
        return self.scaler.transform(data_arcsinh)


class TextProcessor:
    """
    Handles text processing: Dual-Lexical TF-IDF for RF and SBERT Embeddings for MLP.
    """

    def __init__(self):
        self.tfidf_title = TfidfVectorizer(
            max_features=Config.TFIDF_TITLE_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        self.tfidf_body = TfidfVectorizer(
            max_features=Config.TFIDF_BODY_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        # SBERT model loaded lazily or passed in?
        # We'll initialize it here.
        self.sbert_model = SentenceTransformer(
            Config.SBERT_MODEL_NAME, device=Config.DEVICE
        )
        self.is_fitted = False

    def fit(self, train_df):
        """
        Fits TF-IDF vectorizers on training text.
        """
        print("Fitting TF-IDF vectorizers...")
        self.tfidf_title.fit(train_df["request_title"])
        self.tfidf_body.fit(train_df["request_text_edit_aware"])
        self.is_fitted = True

    def transform_tfidf(self, df):
        """
        Generates Dual-Lexical TF-IDF features (Dense).
        """
        if not self.is_fitted:
            raise RuntimeError("TextProcessor must be fitted before transform.")

        title_feats = self.tfidf_title.transform(df["request_title"]).toarray()
        body_feats = self.tfidf_body.transform(df["request_text_edit_aware"]).toarray()

        return np.hstack([title_feats, body_feats])

    def transform_sbert(self, df):
        """
        Generates SBERT embeddings for Title and Body.
        Returns tuple (title_emb, body_emb).
        """
        print("Generating SBERT embeddings...")
        # Batch encoding is handled by SentenceTransformer
        title_emb = self.sbert_model.encode(
            df["request_title"].tolist(),
            batch_size=Config.SBERT_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        body_emb = self.sbert_model.encode(
            df["request_text_edit_aware"].tolist(),
            batch_size=Config.SBERT_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return title_emb, body_emb


class HistoryProcessor:
    """
    Handles user history processing:
    1. Discrete Topic Modeling (K-Means) for RF.
    2. Sequence of Subreddit Embeddings for MLP Attention.
    """

    def __init__(self):
        self.sbert_model = SentenceTransformer(
            Config.SBERT_MODEL_NAME, device=Config.DEVICE
        )
        self.kmeans = KMeans(
            n_clusters=Config.NUM_TOPIC_CLUSTERS,
            random_state=Config.RANDOM_SEED,
            n_init=10,
        )
        self.sub_to_emb = {}
        self.sub_to_cluster = {}
        self.is_fitted = False
        self.max_seq_len = 50  # Fixed sequence length for attention

    def _get_unique_subreddits(self, dfs):
        """
        Extracts all unique subreddits from a list of dataframes.
        """
        unique_subs = set()
        for df in dfs:
            for sub_list in df["requester_subreddits_at_request"]:
                unique_subs.update(sub_list)
        return list(unique_subs)

    def fit(self, train_df, all_dfs):
        """
        Fits K-Means on subreddit embeddings.
        Note: We compute embeddings for ALL unique subreddits in the dataset (train/val/test)
        to handle OOV in test, but we FIT K-Means only on subreddits present in TRAIN
        to avoid data leakage regarding cluster distribution.
        """
        print("Processing subreddit history...")
        all_unique_subs = self._get_unique_subreddits(all_dfs)
        train_unique_subs = self._get_unique_subreddits([train_df])

        # 1. Compute Embeddings for ALL subreddits
        print(f"Encoding {len(all_unique_subs)} unique subreddits...")
        embeddings = self.sbert_model.encode(
            all_unique_subs,
            batch_size=Config.SBERT_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.sub_to_emb = {sub: emb for sub, emb in zip(all_unique_subs, embeddings)}

        # 2. Fit K-Means on TRAIN subreddits only
        train_sub_embeddings = [self.sub_to_emb[sub] for sub in train_unique_subs]
        if len(train_sub_embeddings) < Config.NUM_TOPIC_CLUSTERS:
            # Fallback for very small debug datasets
            print("Warning: Not enough subreddits for K-Means. Reducing clusters.")
            self.kmeans = KMeans(
                n_clusters=max(1, len(train_sub_embeddings)),
                random_state=Config.RANDOM_SEED,
                n_init=10,
            )

        self.kmeans.fit(train_sub_embeddings)

        # 3. Predict clusters for ALL subreddits (so we can handle test data)
        # This is a valid transformation, not leakage, as we use the fitted centroids.
        all_labels = self.kmeans.predict(embeddings)
        self.sub_to_cluster = {
            sub: label for sub, label in zip(all_unique_subs, all_labels)
        }

        self.is_fitted = True

    def transform_topics(self, df):
        """
        Generates discrete topic distribution features for RF.
        Returns (N, K) array where K is num_clusters.
        """
        if not self.is_fitted:
            raise RuntimeError("HistoryProcessor must be fitted before transform.")

        n_samples = len(df)
        n_clusters = self.kmeans.n_clusters
        topic_features = np.zeros((n_samples, n_clusters), dtype=np.float32)

        for i, sub_list in enumerate(df["requester_subreddits_at_request"]):
            if not sub_list:
                continue

            # Count clusters
            counts = np.zeros(n_clusters)
            for sub in sub_list:
                if sub in self.sub_to_cluster:
                    cluster_id = self.sub_to_cluster[sub]
                    counts[cluster_id] += 1

            # Normalize to get ratio
            total = len(sub_list)
            if total > 0:
                topic_features[i] = counts / total

        return topic_features

    def transform_sequences(self, df):
        """
        Generates sequence of embeddings for MLP Attention.
        Returns (N, MaxLen, EmbDim) array.
        """
        if not self.is_fitted:
            raise RuntimeError("HistoryProcessor must be fitted before transform.")

        n_samples = len(df)
        emb_dim = Config.EMBEDDING_DIM
        sequences = np.zeros((n_samples, self.max_seq_len, emb_dim), dtype=np.float32)

        for i, sub_list in enumerate(df["requester_subreddits_at_request"]):
            if not sub_list:
                continue

            # Get embeddings for history
            # Truncate to max_seq_len
            current_subs = sub_list[: self.max_seq_len]
            for t, sub in enumerate(current_subs):
                if sub in self.sub_to_emb:
                    sequences[i, t, :] = self.sub_to_emb[sub]

            # Padding is implicitly zero due to initialization

        return sequences


def run_feature_engineering(load_cached_data=True, debug=Config.DEBUG):
    """
    Main execution function.
    Loads data, fits processors, transforms data, and caches results.
    """
    # 1. Check Cache
    cache_files = {
        "train": os.path.join(Config.WORKING_DIR, "train_features.npz"),
        "val": os.path.join(Config.WORKING_DIR, "val_features.npz"),
        "test": os.path.join(Config.WORKING_DIR, "test_features.npz"),
    }

    if load_cached_data:
        all_exist = all(os.path.exists(f) for f in cache_files.values())
        if all_exist:
            print("Loading features from cache...")
            return (
                np.load(cache_files["train"]),
                np.load(cache_files["val"]),
                np.load(cache_files["test"]),
            )
        else:
            print("Cache incomplete or missing. Generating features...")

    # 2. Load Data
    train_df, val_df, test_df = load_data(load_cached_data=True, debug=debug)

    # 3. Initialize Processors
    meta_proc = MetadataProcessor()
    text_proc = TextProcessor()
    hist_proc = HistoryProcessor()

    # 4. Fit Processors
    print("Fitting Metadata Processor...")
    meta_proc.fit(train_df)

    print("Fitting Text Processor...")
    text_proc.fit(train_df)

    print("Fitting History Processor...")
    hist_proc.fit(train_df, [train_df, val_df, test_df])

    # 5. Transform and Bundle
    datasets = {"train": train_df, "val": val_df, "test": test_df}

    results = {}

    for split_name, df in datasets.items():
        print(f"Transforming {split_name} set...")

        # Metadata
        rf_meta = meta_proc.transform_rf(df)
        mlp_meta = meta_proc.transform_mlp(df)

        # Text
        rf_tfidf = text_proc.transform_tfidf(df)
        mlp_title_emb, mlp_body_emb = text_proc.transform_sbert(df)

        # History
        rf_topics = hist_proc.transform_topics(df)
        mlp_hist_emb = hist_proc.transform_sequences(df)

        # Target (if available)
        y = np.array([])
        if "requester_received_pizza" in df.columns:
            y = df["requester_received_pizza"].astype(int).values

        # IDs (for submission)
        ids = df["request_id"].values

        # Pack into dictionary
        data_dict = {
            "rf_meta": rf_meta,
            "rf_tfidf": rf_tfidf,
            "rf_topics": rf_topics,
            "mlp_meta": mlp_meta,
            "mlp_title_emb": mlp_title_emb,
            "mlp_body_emb": mlp_body_emb,
            "mlp_hist_emb": mlp_hist_emb,
            "y": y,
            "ids": ids,
        }

        results[split_name] = data_dict

        # Cache to disk
        print(f"Saving {split_name} features to {cache_files[split_name]}...")
        np.savez(cache_files[split_name], **data_dict)

    return (
        np.load(cache_files["train"]),
        np.load(cache_files["val"]),
        np.load(cache_files["test"]),
    )
