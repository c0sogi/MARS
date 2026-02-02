import os
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from library.utils import load_or_compute, Timer


class MetadataExtractor:
    """
    Extracts and transforms numerical and meta-features from the dataset.
    Handles raw magnitudes, engineered ratios, and text meta-features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
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
        self.impute_values = None

    def _compute_features(self, df):
        """
        Computes raw and engineered features for a single dataframe.
        """
        df = df.copy()

        # Handle missing text
        df["request_text_edit_aware"] = df["request_text_edit_aware"].fillna("")
        df["request_title"] = df["request_title"].fillna("")

        # --- Text Meta-Features ---
        # Character and Word Counts
        df["text_char_count"] = df["request_text_edit_aware"].apply(len)
        df["text_word_count"] = df["request_text_edit_aware"].apply(
            lambda x: len(str(x).split())
        )
        df["title_char_count"] = df["request_title"].apply(len)
        df["title_word_count"] = df["request_title"].apply(
            lambda x: len(str(x).split())
        )

        # Caps Ratio
        def get_caps_ratio(text):
            if len(text) == 0:
                return 0.0
            return sum(1 for c in text if c.isupper()) / len(text)

        df["text_caps_ratio"] = df["request_text_edit_aware"].apply(get_caps_ratio)
        df["title_caps_ratio"] = df["request_title"].apply(get_caps_ratio)

        # --- Engineered Ratios ---
        epsilon = 1e-6
        # Upvote Ratio: (up - down) / (up + down)
        # Note: We only have diff and sum available directly.
        # Ratio = diff / sum
        df["upvote_ratio"] = df["requester_upvotes_minus_downvotes_at_request"] / (
            df["requester_upvotes_plus_downvotes_at_request"] + epsilon
        )

        # Activity Ratios
        df["raop_comment_ratio"] = df[
            "requester_number_of_comments_in_raop_at_request"
        ] / (df["requester_number_of_comments_at_request"] + epsilon)
        df["raop_post_ratio"] = df["requester_number_of_posts_on_raop_at_request"] / (
            df["requester_number_of_posts_at_request"] + epsilon
        )

        # Select relevant columns
        meta_cols = [
            "text_char_count",
            "text_word_count",
            "title_char_count",
            "title_word_count",
            "text_caps_ratio",
            "title_caps_ratio",
            "upvote_ratio",
            "raop_comment_ratio",
            "raop_post_ratio",
        ]

        # Ensure raw numeric cols exist in this dataframe
        available_numeric = [c for c in self.numeric_cols if c in df.columns]

        final_df = df[available_numeric + meta_cols].copy()

        return final_df

    def process(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Generates processed metadata features for all splits.
        Performs median imputation.
        """
        # Cache keys
        train_key = "metadata_features_train.parquet"
        val_key = "metadata_features_val.parquet"
        test_key = "metadata_features_test.parquet"

        # Compute or Load
        train_feats = load_or_compute(
            train_key,
            lambda: self._compute_features(df_train),
            load_cached_data,
            file_type="parquet",
        )
        val_feats = load_or_compute(
            val_key,
            lambda: self._compute_features(df_val),
            load_cached_data,
            file_type="parquet",
        )
        test_feats = load_or_compute(
            test_key,
            lambda: self._compute_features(df_test),
            load_cached_data,
            file_type="parquet",
        )

        # Imputation (Fit on Train)
        if self.impute_values is None:
            self.impute_values = train_feats.median()

        train_feats = train_feats.fillna(self.impute_values)
        val_feats = val_feats.fillna(self.impute_values)
        test_feats = test_feats.fillna(self.impute_values)

        return train_feats, val_feats, test_feats

    def get_scaled_features(self, train_feats, val_feats, test_feats):
        """
        Applies Arcsinh transformation and StandardScaler for MLP input.
        """
        # Arcsinh transform (handles skew and zeros well)
        train_log = np.arcsinh(train_feats)
        val_log = np.arcsinh(val_feats)
        test_log = np.arcsinh(test_feats)

        # Fit Scaler on Train
        self.scaler.fit(train_log)

        # Transform
        X_train = self.scaler.transform(train_log).astype(np.float32)
        X_val = self.scaler.transform(val_log).astype(np.float32)
        X_test = self.scaler.transform(test_log).astype(np.float32)

        return X_train, X_val, X_test


class TopKSubredditEncoder:
    """
    Encodes the presence of the top-K most frequent subreddits as binary flags.
    """

    def __init__(self, k=50):
        self.k = k
        self.top_subreddits = None

    def _parse_subreddits(self, series):
        """Parses string representation of lists into actual lists."""

        def parse(x):
            try:
                if isinstance(x, list):
                    return x
                return ast.literal_eval(x)
            except:
                return []

        return series.apply(parse)

    def fit(self, df):
        """Identifies top K subreddits from the dataframe."""
        subs = self._parse_subreddits(df["requester_subreddits_at_request"])
        all_subs = [s for sublist in subs for s in sublist]
        if not all_subs:
            self.top_subreddits = []
        else:
            counts = pd.Series(all_subs).value_counts()
            self.top_subreddits = counts.head(self.k).index.tolist()
        return self.top_subreddits

    def transform(self, df):
        """Creates binary indicator matrix for top K subreddits."""
        if self.top_subreddits is None:
            raise ValueError("Encoder not fitted. Call fit() first.")

        subs = self._parse_subreddits(df["requester_subreddits_at_request"])

        # Efficient construction
        data = []
        for sublist in subs:
            # Create a dict for this row
            row = {f"sub_{s}": 1 for s in sublist if s in self.top_subreddits}
            data.append(row)

        feat_df = pd.DataFrame(data, index=df.index)

        # Ensure all columns exist and are sorted
        for s in self.top_subreddits:
            col = f"sub_{s}"
            if col not in feat_df.columns:
                feat_df[col] = 0

        # Sort columns for consistency
        cols = sorted([f"sub_{s}" for s in self.top_subreddits])
        feat_df = feat_df[cols].fillna(0).astype(int)

        return feat_df

    def process(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Fits on train, transforms all splits. Handles caching.
        """
        # Always fit on the provided training data to ensure consistency
        self.fit(df_train)

        train_key = "topk_features_train.parquet"
        val_key = "topk_features_val.parquet"
        test_key = "topk_features_test.parquet"

        train_feats = load_or_compute(
            train_key,
            lambda: self.transform(df_train),
            load_cached_data,
            file_type="parquet",
        )
        val_feats = load_or_compute(
            val_key,
            lambda: self.transform(df_val),
            load_cached_data,
            file_type="parquet",
        )
        test_feats = load_or_compute(
            test_key,
            lambda: self.transform(df_test),
            load_cached_data,
            file_type="parquet",
        )

        return train_feats, val_feats, test_feats


class TextProcessor:
    """
    Handles text vectorization (TF-IDF) and embedding (SBERT).
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        self.sbert = None  # Lazy loading to save resources if not used
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_text(self, df):
        t = df["request_title"].fillna("")
        b = df["request_text_edit_aware"].fillna("")
        return t + " " + b

    def process_tfidf(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Generates TF-IDF features. Fits on Train, transforms all.
        Returns dense numpy arrays.
        """
        cache_dir = "./working/idea_25/"
        train_path = os.path.join(cache_dir, "tfidf_train.npy")
        val_path = os.path.join(cache_dir, "tfidf_val.npy")
        test_path = os.path.join(cache_dir, "tfidf_test.npy")

        all_exist = all(os.path.exists(p) for p in [train_path, val_path, test_path])

        if load_cached_data and all_exist:
            print("Loading cached TF-IDF data...")
            return np.load(train_path), np.load(val_path), np.load(test_path)

        print("Computing TF-IDF features...")
        train_text = self._get_text(df_train)
        val_text = self._get_text(df_val)
        test_text = self._get_text(df_test)

        self.tfidf.fit(train_text)

        X_train = self.tfidf.transform(train_text).toarray().astype(np.float32)
        X_val = self.tfidf.transform(val_text).toarray().astype(np.float32)
        X_test = self.tfidf.transform(test_text).toarray().astype(np.float32)

        os.makedirs(cache_dir, exist_ok=True)
        np.save(train_path, X_train)
        np.save(val_path, X_val)
        np.save(test_path, X_test)

        return X_train, X_val, X_test

    def _load_sbert(self):
        if self.sbert is None:
            print(f"Loading SBERT model on {self.device}...")
            self.sbert = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

    def process_sbert_request(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Generates SBERT embeddings for the request text (Title + Body).
        """
        self._load_sbert()

        def compute(df):
            text = self._get_text(df).tolist()
            return self.sbert.encode(
                text, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )

        train_key = "sbert_request_train.npy"
        val_key = "sbert_request_val.npy"
        test_key = "sbert_request_test.npy"

        X_train = load_or_compute(
            train_key, lambda: compute(df_train), load_cached_data, file_type="npy"
        )
        X_val = load_or_compute(
            val_key, lambda: compute(df_val), load_cached_data, file_type="npy"
        )
        X_test = load_or_compute(
            test_key, lambda: compute(df_test), load_cached_data, file_type="npy"
        )

        return X_train, X_val, X_test

    def process_sbert_history(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Generates sequence of SBERT embeddings for user subreddit history.
        Output shape: (N, 20, 384).
        """
        self._load_sbert()
        max_len = 20
        embedding_dim = 384

        def parse_subs(series):
            def parse(x):
                try:
                    if isinstance(x, list):
                        return x
                    return ast.literal_eval(x)
                except:
                    return []

            return series.apply(parse)

        def compute(df):
            # 1. Parse all subreddits
            subs_series = parse_subs(df["requester_subreddits_at_request"])

            # 2. Identify unique subreddits to batch encode
            all_subs = set()
            for s_list in subs_series:
                all_subs.update(s_list)

            unique_subs = list(all_subs)
            sub_map = {}

            if unique_subs:
                # Batch encode unique subs
                print(f"Encoding {len(unique_subs)} unique subreddits for history...")
                sub_embeddings = self.sbert.encode(
                    unique_subs,
                    batch_size=128,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                sub_map = {name: emb for name, emb in zip(unique_subs, sub_embeddings)}

            # 3. Construct tensor
            N = len(df)
            output = np.zeros((N, max_len, embedding_dim), dtype=np.float32)

            for i, s_list in enumerate(subs_series):
                # Take first 20 subreddits (assuming relevance or chronological order)
                current_subs = s_list[:max_len]

                for j, sub_name in enumerate(current_subs):
                    if sub_name in sub_map:
                        output[i, j, :] = sub_map[sub_name]

            return output

        train_key = "sbert_history_train.npy"
        val_key = "sbert_history_val.npy"
        test_key = "sbert_history_test.npy"

        X_train = load_or_compute(
            train_key, lambda: compute(df_train), load_cached_data, file_type="npy"
        )
        X_val = load_or_compute(
            val_key, lambda: compute(df_val), load_cached_data, file_type="npy"
        )
        X_test = load_or_compute(
            test_key, lambda: compute(df_test), load_cached_data, file_type="npy"
        )

        return X_train, X_val, X_test
