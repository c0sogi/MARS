import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp

from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    LEXICONS,
    SBERT_MODEL_NAME,
    RANDOM_STATE,
)
from library.utils import set_seed


class Preprocessor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            stop_words="english",
            binary=True,  # Focus on presence/absence for specific triggers
        )
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy="constant", fill_value=0)
        self.sbert_model = None  # Lazy load

    def _load_sbert(self):
        if self.sbert_model is None:
            # Load SBERT model on demand
            self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
            if hasattr(self.sbert_model, "to"):
                import torch

                if torch.cuda.is_available():
                    self.sbert_model = self.sbert_model.to("cuda")

    def get_feature_intersection(self, df_train, df_test):
        """
        Identifies common numeric columns between train and test, removing leakage.
        """
        # Identify numeric columns in both
        train_cols = set(df_train.select_dtypes(include=[np.number, bool]).columns)
        test_cols = set(df_test.select_dtypes(include=[np.number, bool]).columns)

        common_cols = list(train_cols.intersection(test_cols))

        # Filter out leakage and identifiers
        final_cols = []
        for col in common_cols:
            if col.endswith("_at_retrieval"):
                continue
            if (
                "timestamp" in col
            ):  # Timestamps can be tricky, often correlated with split
                continue
            if col in ["requester_received_pizza", "request_id"]:
                continue
            final_cols.append(col)

        return sorted(final_cols)

    def extract_lexicon_features(self, df, text_col):
        """
        Computes density of domain-specific keywords.
        """
        features = pd.DataFrame(index=df.index)

        # Pre-calculate word counts to avoid division by zero
        # Fill NaNs in text
        texts = df[text_col].fillna("").astype(str).str.lower()
        word_counts = texts.apply(lambda x: len(x.split()))
        word_counts = word_counts.replace(0, 1)  # Avoid div by zero

        for category, keywords in LEXICONS.items():
            # Create regex pattern for the category
            pattern = "|".join([re.escape(k) for k in keywords])

            # Count matches
            match_counts = texts.apply(lambda x: len(re.findall(pattern, x)))

            # Calculate density
            features[f"lexicon_density_{category}"] = match_counts / word_counts

        return features

    def extract_meta_features(self, df, text_col):
        """
        Extracts structural text features.
        """
        features = pd.DataFrame(index=df.index)
        texts = df[text_col].fillna("").astype(str)

        features["text_len_char"] = texts.apply(len)
        features["text_len_word"] = texts.apply(lambda x: len(x.split()))

        # Caps ratio (shouting or emphasis)
        def get_caps_ratio(s):
            if len(s) == 0:
                return 0.0
            return sum(1 for c in s if c.isupper()) / len(s)

        features["text_caps_ratio"] = texts.apply(get_caps_ratio)

        return features

    def engineer_ratios(self, df):
        """
        Creates behavioral ratio features.
        """
        features = pd.DataFrame(index=df.index)

        # Upvote ratio
        up = df.get("requester_upvotes_plus_downvotes_at_request", 0)
        diff = df.get("requester_upvotes_minus_downvotes_at_request", 0)
        # diff = up - down, sum = up + down -> down = (sum - diff) / 2
        # But we can just use diff/sum as a proxy for quality
        features["upvote_ratio_proxy"] = diff / (up + 1.0)

        # Comments per post
        n_posts = df.get("requester_number_of_posts_at_request", 0)
        n_comments = df.get("requester_number_of_comments_at_request", 0)
        features["comments_per_post"] = n_comments / (n_posts + 1.0)

        # RAOP activity ratio
        n_posts_raop = df.get("requester_number_of_posts_on_raop_at_request", 0)
        features["raop_post_ratio"] = n_posts_raop / (n_posts + 1.0)

        return features

    def process_text_tfidf(self, train_text, val_text, test_text):
        """
        Generates TF-IDF features.
        """
        print("Generating TF-IDF features...")
        X_train = self.tfidf.fit_transform(train_text)
        X_val = self.tfidf.transform(val_text)
        X_test = self.tfidf.transform(test_text)
        return X_train, X_val, X_test

    def process_text_sbert(self, train_text, val_text, test_text):
        """
        Generates SBERT embeddings.
        """
        print("Generating SBERT embeddings...")
        self._load_sbert()

        # Encode in batches
        X_train = self.sbert_model.encode(
            train_text.tolist(), batch_size=32, show_progress_bar=False
        )
        X_val = self.sbert_model.encode(
            val_text.tolist(), batch_size=32, show_progress_bar=False
        )
        X_test = self.sbert_model.encode(
            test_text.tolist(), batch_size=32, show_progress_bar=False
        )

        return X_train, X_val, X_test

    def run(self, load_cached_data=True):
        """
        Main execution method.
        Returns dictionary containing processed datasets for RF and MLP.
        """
        set_seed()

        # Define cache file paths
        files = {
            "rf_train_tab": "rf_train_tab.npy",
            "rf_val_tab": "rf_val_tab.npy",
            "rf_test_tab": "rf_test_tab.npy",
            "rf_train_tfidf": "rf_train_tfidf.npz",
            "rf_val_tfidf": "rf_val_tfidf.npz",
            "rf_test_tfidf": "rf_test_tfidf.npz",
            "mlp_train_tab": "mlp_train_tab.npy",
            "mlp_val_tab": "mlp_val_tab.npy",
            "mlp_test_tab": "mlp_test_tab.npy",
            "mlp_train_sbert": "mlp_train_sbert.npy",
            "mlp_val_sbert": "mlp_val_sbert.npy",
            "mlp_test_sbert": "mlp_test_sbert.npy",
            "y_train": "y_train.npy",
            "y_val": "y_val.npy",
            "test_ids": "test_ids.npy",
        }

        # Check if all files exist
        all_exist = all(
            os.path.exists(os.path.join(CACHE_DIR, f)) for f in files.values()
        )

        if load_cached_data and all_exist:
            print("Loading cached data...")
            data = {}
            data["rf_train_tab"] = np.load(
                os.path.join(CACHE_DIR, files["rf_train_tab"])
            )
            data["rf_val_tab"] = np.load(os.path.join(CACHE_DIR, files["rf_val_tab"]))
            data["rf_test_tab"] = np.load(os.path.join(CACHE_DIR, files["rf_test_tab"]))

            data["rf_train_tfidf"] = sp.load_npz(
                os.path.join(CACHE_DIR, files["rf_train_tfidf"])
            )
            data["rf_val_tfidf"] = sp.load_npz(
                os.path.join(CACHE_DIR, files["rf_val_tfidf"])
            )
            data["rf_test_tfidf"] = sp.load_npz(
                os.path.join(CACHE_DIR, files["rf_test_tfidf"])
            )

            data["mlp_train_tab"] = np.load(
                os.path.join(CACHE_DIR, files["mlp_train_tab"])
            )
            data["mlp_val_tab"] = np.load(os.path.join(CACHE_DIR, files["mlp_val_tab"]))
            data["mlp_test_tab"] = np.load(
                os.path.join(CACHE_DIR, files["mlp_test_tab"])
            )

            data["mlp_train_sbert"] = np.load(
                os.path.join(CACHE_DIR, files["mlp_train_sbert"])
            )
            data["mlp_val_sbert"] = np.load(
                os.path.join(CACHE_DIR, files["mlp_val_sbert"])
            )
            data["mlp_test_sbert"] = np.load(
                os.path.join(CACHE_DIR, files["mlp_test_sbert"])
            )

            data["y_train"] = np.load(os.path.join(CACHE_DIR, files["y_train"]))
            data["y_val"] = np.load(os.path.join(CACHE_DIR, files["y_val"]))
            data["test_ids"] = np.load(
                os.path.join(CACHE_DIR, files["test_ids"]), allow_pickle=True
            )

            return data

        print("Processing data from scratch...")

        # Load Raw Data
        df_train = pd.read_csv(TRAIN_PATH)
        df_val = pd.read_csv(VAL_PATH)
        df_test = pd.read_csv(TEST_PATH)

        # Target
        y_train = df_train["requester_received_pizza"].astype(int).values
        y_val = df_val["requester_received_pizza"].astype(int).values
        test_ids = df_test["request_id"].values

        # Text Column Selection
        text_col = "request_text_edit_aware"

        # 1. Tabular Feature Engineering (Full Spectrum)
        # Intersection of raw numeric features
        base_cols = self.get_feature_intersection(df_train, df_test)

        def process_tabular(df):
            # Base numeric features
            tab = df[base_cols].copy()

            # Ratios
            ratios = self.engineer_ratios(df)
            tab = pd.concat([tab, ratios], axis=1)

            # Meta features
            meta = self.extract_meta_features(df, text_col)
            tab = pd.concat([tab, meta], axis=1)

            # Lexicon features
            lex = self.extract_lexicon_features(df, text_col)
            tab = pd.concat([tab, lex], axis=1)

            return tab

        print("Engineering tabular features...")
        X_train_tab_raw = process_tabular(df_train)
        X_val_tab_raw = process_tabular(df_val)
        X_test_tab_raw = process_tabular(df_test)

        # Impute NaNs (RF handles them sometimes, but safer to impute for consistency)
        # Using fit on train, transform on val/test
        cols = X_train_tab_raw.columns
        X_train_tab_raw = pd.DataFrame(
            self.imputer.fit_transform(X_train_tab_raw), columns=cols
        )
        X_val_tab_raw = pd.DataFrame(
            self.imputer.transform(X_val_tab_raw), columns=cols
        )
        X_test_tab_raw = pd.DataFrame(
            self.imputer.transform(X_test_tab_raw), columns=cols
        )

        # 2. Tabular Scaling for MLP
        # Use arcsinh to handle skew/negatives (Cite Lesson 24) and MinMaxScaler to prevent explosion
        print("Scaling tabular features for MLP...")
        X_train_tab_scaled = np.arcsinh(X_train_tab_raw)
        X_val_tab_scaled = np.arcsinh(X_val_tab_raw)
        X_test_tab_scaled = np.arcsinh(X_test_tab_raw)

        X_train_tab_scaled = self.scaler.fit_transform(X_train_tab_scaled)
        X_val_tab_scaled = self.scaler.transform(X_val_tab_scaled)
        X_test_tab_scaled = self.scaler.transform(X_test_tab_scaled)

        # Convert raw to numpy for RF
        X_train_tab_raw = X_train_tab_raw.values.astype(np.float32)
        X_val_tab_raw = X_val_tab_raw.values.astype(np.float32)
        X_test_tab_raw = X_test_tab_raw.values.astype(np.float32)

        # 3. Text Processing
        # Fill NaNs in text
        train_text = df_train[text_col].fillna("").astype(str)
        val_text = df_val[text_col].fillna("").astype(str)
        test_text = df_test[text_col].fillna("").astype(str)

        # TF-IDF (Sparse) for RF
        X_train_tfidf, X_val_tfidf, X_test_tfidf = self.process_text_tfidf(
            train_text, val_text, test_text
        )

        # SBERT (Dense) for MLP
        X_train_sbert, X_val_sbert, X_test_sbert = self.process_text_sbert(
            train_text, val_text, test_text
        )

        # Save to Cache
        print("Saving processed data to cache...")
        np.save(os.path.join(CACHE_DIR, files["rf_train_tab"]), X_train_tab_raw)
        np.save(os.path.join(CACHE_DIR, files["rf_val_tab"]), X_val_tab_raw)
        np.save(os.path.join(CACHE_DIR, files["rf_test_tab"]), X_test_tab_raw)

        sp.save_npz(os.path.join(CACHE_DIR, files["rf_train_tfidf"]), X_train_tfidf)
        sp.save_npz(os.path.join(CACHE_DIR, files["rf_val_tfidf"]), X_val_tfidf)
        sp.save_npz(os.path.join(CACHE_DIR, files["rf_test_tfidf"]), X_test_tfidf)

        np.save(os.path.join(CACHE_DIR, files["mlp_train_tab"]), X_train_tab_scaled)
        np.save(os.path.join(CACHE_DIR, files["mlp_val_tab"]), X_val_tab_scaled)
        np.save(os.path.join(CACHE_DIR, files["mlp_test_tab"]), X_test_tab_scaled)

        np.save(os.path.join(CACHE_DIR, files["mlp_train_sbert"]), X_train_sbert)
        np.save(os.path.join(CACHE_DIR, files["mlp_val_sbert"]), X_val_sbert)
        np.save(os.path.join(CACHE_DIR, files["mlp_test_sbert"]), X_test_sbert)

        np.save(os.path.join(CACHE_DIR, files["y_train"]), y_train)
        np.save(os.path.join(CACHE_DIR, files["y_val"]), y_val)
        np.save(os.path.join(CACHE_DIR, files["test_ids"]), test_ids)

        return {
            "rf_train_tab": X_train_tab_raw,
            "rf_val_tab": X_val_tab_raw,
            "rf_test_tab": X_test_tab_raw,
            "rf_train_tfidf": X_train_tfidf,
            "rf_val_tfidf": X_val_tfidf,
            "rf_test_tfidf": X_test_tfidf,
            "mlp_train_tab": X_train_tab_scaled,
            "mlp_val_tab": X_val_tab_scaled,
            "mlp_test_tab": X_test_tab_scaled,
            "mlp_train_sbert": X_train_sbert,
            "mlp_val_sbert": X_val_sbert,
            "mlp_test_sbert": X_test_sbert,
            "y_train": y_train,
            "y_val": y_val,
            "test_ids": test_ids,
        }
