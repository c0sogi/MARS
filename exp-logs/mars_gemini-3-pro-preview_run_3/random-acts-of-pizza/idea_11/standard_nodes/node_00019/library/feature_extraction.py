import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config


class LexicalTransformer:
    """
    Generates sparse TF-IDF features from request text.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=Config.TEXT_TFIDF_MAX_FEATURES,
            ngram_range=Config.TEXT_TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )

    def fit(self, text_series):
        print("Fitting Lexical TF-IDF...")
        self.vectorizer.fit(text_series.fillna(""))
        return self

    def transform(self, text_series):
        return self.vectorizer.transform(text_series.fillna(""))


class BehavioralTransformer:
    """
    Generates both sparse TF-IDF features and dense SVD latent features from subreddit history.
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=Config.SUBREDDIT_TFIDF_MAX_FEATURES,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",  # Simple token pattern for subreddit names
        )
        self.svd = TruncatedSVD(
            n_components=Config.SUBREDDIT_SVD_COMPONENTS,
            random_state=Config.RANDOM_SEED,
        )

    def fit(self, subreddit_series):
        print("Fitting Behavioral TF-IDF and SVD...")
        # Fill NA and ensure strings
        subs = subreddit_series.fillna("").astype(str)
        X_sparse = self.tfidf.fit_transform(subs)
        self.svd.fit(X_sparse)
        return self

    def transform(self, subreddit_series):
        subs = subreddit_series.fillna("").astype(str)
        X_sparse = self.tfidf.transform(subs)
        X_dense = self.svd.transform(X_sparse)
        return X_sparse, X_dense


class SemanticTransformer:
    """
    Generates dense embeddings using SBERT.
    """

    def __init__(self):
        self.model_name = Config.SBERT_MODEL_NAME
        self.model = None

    def fit(self, X=None, y=None):
        # SBERT is pre-trained, no fitting required on our data
        print(f"Loading SBERT model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, text_series):
        print("Generating Semantic Embeddings...")
        texts = text_series.fillna("").astype(str).tolist()
        # Encode returns a numpy array
        embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
        return embeddings


class MetaFeatureTransformer:
    """
    Extracts, imputes, and scales numerical metadata and text complexity features.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_names = []

        # Hardcoded list of numerical columns to use from metadata
        self.numerical_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
        ]

    def _extract_text_complexity(self, df):
        """
        Computes simple text complexity features: Word Count, Sentence Count.
        """
        texts = df[Config.TEXT_COL].fillna("").astype(str)

        # Word count
        word_counts = texts.apply(lambda x: len(x.split()))

        # Sentence count (simple regex split)
        sent_counts = texts.apply(lambda x: len(re.split(r"[.!?]+", x)) - 1 if x else 0)
        sent_counts = sent_counts.replace(0, 1)  # Avoid 0 division issues if used later

        return pd.DataFrame({"word_count": word_counts, "sentence_count": sent_counts})

    def fit(self, df):
        print("Fitting Meta Features...")
        # Extract numerical features
        meta_df = df[self.numerical_cols].copy()

        # Extract text complexity
        complexity_df = self._extract_text_complexity(df)

        # Combine
        X_combined = pd.concat([meta_df, complexity_df], axis=1)

        # Fit imputer and scaler
        self.imputer.fit(X_combined)
        self.scaler.fit(self.imputer.transform(X_combined))

        self.feature_names = X_combined.columns.tolist()
        return self

    def transform(self, df):
        meta_df = df[self.numerical_cols].copy()
        complexity_df = self._extract_text_complexity(df)
        X_combined = pd.concat([meta_df, complexity_df], axis=1)

        X_imputed = self.imputer.transform(X_combined)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled


class FeatureManager:
    """
    Orchestrates feature extraction, caching, and data preparation.
    """

    def __init__(self):
        self.lexical_transformer = LexicalTransformer()
        self.behavioral_transformer = BehavioralTransformer()
        self.semantic_transformer = SemanticTransformer()
        self.meta_transformer = MetaFeatureTransformer()
        self.is_fitted = False

    def fit(self, train_df):
        print("Fitting all transformers on training data...")
        self.lexical_transformer.fit(train_df[Config.TEXT_COL])
        self.behavioral_transformer.fit(train_df[Config.SUBREDDIT_COL])
        self.semantic_transformer.fit()
        self.meta_transformer.fit(train_df)
        self.is_fitted = True

    def transform(self, df, split_name="data"):
        if not self.is_fitted:
            raise RuntimeError("FeatureManager must be fitted before transform.")

        print(f"Transforming {split_name}...")

        # 1. Lexical (Sparse)
        X_lexical = self.lexical_transformer.transform(df[Config.TEXT_COL])

        # 2. Behavioral (Sparse & Dense)
        X_beh_sparse, X_beh_dense = self.behavioral_transformer.transform(
            df[Config.SUBREDDIT_COL]
        )

        # 3. Semantic (Dense)
        X_semantic = self.semantic_transformer.transform(df[Config.TEXT_COL])

        # 4. Meta (Dense)
        X_meta = self.meta_transformer.transform(df)

        # 5. Unified Dense View
        # Concatenate: Semantic (384) + Behavioral SVD (20) + Meta (~8)
        X_dense = np.hstack([X_semantic, X_beh_dense, X_meta])

        return {
            "X_lexical": X_lexical,
            "X_behavioral": X_beh_sparse,
            "X_dense": X_dense,
        }

    def extract_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main entry point. Handles caching and feature generation.
        """
        # Define cache paths
        cache_files = {
            "train": {
                "lexical": os.path.join(Config.WORKING_DIR, "X_train_lexical.npz"),
                "behavioral": os.path.join(
                    Config.WORKING_DIR, "X_train_behavioral.npz"
                ),
                "dense": os.path.join(Config.WORKING_DIR, "X_train_dense.npy"),
                "y": os.path.join(Config.WORKING_DIR, "y_train.npy"),
            },
            "val": {
                "lexical": os.path.join(Config.WORKING_DIR, "X_val_lexical.npz"),
                "behavioral": os.path.join(Config.WORKING_DIR, "X_val_behavioral.npz"),
                "dense": os.path.join(Config.WORKING_DIR, "X_val_dense.npy"),
                "y": os.path.join(Config.WORKING_DIR, "y_val.npy"),
            },
            "test": {
                "lexical": os.path.join(Config.WORKING_DIR, "X_test_lexical.npz"),
                "behavioral": os.path.join(Config.WORKING_DIR, "X_test_behavioral.npz"),
                "dense": os.path.join(Config.WORKING_DIR, "X_test_dense.npy"),
            },
        }

        # Check if cache exists
        cache_exists = True
        if load_cached_data:
            for split in ["train", "val", "test"]:
                for key, path in cache_files[split].items():
                    if not os.path.exists(path):
                        cache_exists = False
                        break
        else:
            cache_exists = False

        if cache_exists and load_cached_data:
            print("Loading features from cache...")
            data = {}

            # Load Train
            data["X_train_lexical"] = sparse.load_npz(cache_files["train"]["lexical"])
            data["X_train_behavioral"] = sparse.load_npz(
                cache_files["train"]["behavioral"]
            )
            data["X_train_dense"] = np.load(cache_files["train"]["dense"])
            data["y_train"] = np.load(cache_files["train"]["y"])

            # Load Val
            data["X_val_lexical"] = sparse.load_npz(cache_files["val"]["lexical"])
            data["X_val_behavioral"] = sparse.load_npz(cache_files["val"]["behavioral"])
            data["X_val_dense"] = np.load(cache_files["val"]["dense"])
            data["y_val"] = np.load(cache_files["val"]["y"])

            # Load Test
            data["X_test_lexical"] = sparse.load_npz(cache_files["test"]["lexical"])
            data["X_test_behavioral"] = sparse.load_npz(
                cache_files["test"]["behavioral"]
            )
            data["X_test_dense"] = np.load(cache_files["test"]["dense"])

            return data

        # If not cached, compute
        print("Computing features from scratch...")

        # Fit on Train
        self.fit(train_df)

        # Transform all
        train_feats = self.transform(train_df, "Train")
        val_feats = self.transform(val_df, "Val")
        test_feats = self.transform(test_df, "Test")

        # Extract Targets
        y_train = train_df[Config.TARGET_COL].values
        y_val = val_df[Config.TARGET_COL].values

        # Save to Cache
        print("Saving features to cache...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Save Train
        sparse.save_npz(cache_files["train"]["lexical"], train_feats["X_lexical"])
        sparse.save_npz(cache_files["train"]["behavioral"], train_feats["X_behavioral"])
        np.save(cache_files["train"]["dense"], train_feats["X_dense"])
        np.save(cache_files["train"]["y"], y_train)

        # Save Val
        sparse.save_npz(cache_files["val"]["lexical"], val_feats["X_lexical"])
        sparse.save_npz(cache_files["val"]["behavioral"], val_feats["X_behavioral"])
        np.save(cache_files["val"]["dense"], val_feats["X_dense"])
        np.save(cache_files["val"]["y"], y_val)

        # Save Test
        sparse.save_npz(cache_files["test"]["lexical"], test_feats["X_lexical"])
        sparse.save_npz(cache_files["test"]["behavioral"], test_feats["X_behavioral"])
        np.save(cache_files["test"]["dense"], test_feats["X_dense"])

        # Construct return dictionary
        data = {
            "X_train_lexical": train_feats["X_lexical"],
            "X_train_behavioral": train_feats["X_behavioral"],
            "X_train_dense": train_feats["X_dense"],
            "y_train": y_train,
            "X_val_lexical": val_feats["X_lexical"],
            "X_val_behavioral": val_feats["X_behavioral"],
            "X_val_dense": val_feats["X_dense"],
            "y_val": y_val,
            "X_test_lexical": test_feats["X_lexical"],
            "X_test_behavioral": test_feats["X_behavioral"],
            "X_test_dense": test_feats["X_dense"],
        }

        return data
