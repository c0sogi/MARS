import os
import numpy as np
import pandas as pd
import scipy.sparse
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config


class FeaturePipeline:
    def __init__(self):
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # Initialize Transformers

        # 1. Lexical (Text) Vectorizer
        self.lexical_vectorizer = TfidfVectorizer(**Config.TFIDF_PARAMS)

        # 2. Community (Subreddit) Vectorizer
        # We override token_pattern to capture subreddit names which might contain underscores/alphanumeric
        # We use a simpler config for subreddits, but stick to TF-IDF
        self.community_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            min_df=2,
            max_features=Config.SUBREDDIT_VOCAB_SIZE,
            token_pattern=r"(?u)\b\w+\b",
            stop_words=None,  # Subreddit names are not stop words
        )

        # 3. Metadata Preprocessors
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # 4. Semantic Model
        # Loaded lazily or in fit to avoid overhead if cached
        self.embedding_model = None

    def _get_text_data(self, df):
        """Extracts combined text for lexical and semantic processing."""
        # Assumes 'text_combined' is already created by data_processing.py
        if "text_combined" not in df.columns:
            # Fallback if not present (though pipeline should ensure it)
            title = df["request_title"].fillna("").astype(str)
            body = df["request_text_edit_aware"].fillna("").astype(str)
            return title + " " + body
        return df["text_combined"].fillna("").astype(str)

    def _get_community_data(self, df):
        """Extracts subreddit history as space-separated strings."""
        col = Config.SUBREDDIT_COL
        if col not in df.columns:
            return pd.Series(["" for _ in range(len(df))])

        # Convert list of subreddits to space-separated string
        return df[col].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

    def _get_metadata_data(self, df):
        """Extracts numerical metadata."""
        # Select columns
        meta_cols = [c for c in Config.METADATA_COLS if c in df.columns]
        return df[meta_cols]

    def _load_embedding_model(self):
        if self.embedding_model is None:
            # Suppress verbose output
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

    def fit_transform(self, df_train):
        """Fits transformers on training data and returns feature dictionary."""
        print("Fitting feature extractors...")

        # 1. Lexical
        text_train = self._get_text_data(df_train)
        X_train_lexical_sparse = self.lexical_vectorizer.fit_transform(text_train)

        # 2. Community
        community_train = self._get_community_data(df_train)
        X_train_community_sparse = self.community_vectorizer.fit_transform(
            community_train
        )

        # 3. Metadata
        meta_train = self._get_metadata_data(df_train)
        meta_train_imputed = self.imputer.fit_transform(meta_train)
        X_train_meta_dense = self.scaler.fit_transform(meta_train_imputed)

        # 4. Semantic
        self._load_embedding_model()
        # Encode returns numpy array
        X_train_semantic_dense = self.embedding_model.encode(
            text_train.tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Save fitted transformers
        joblib.dump(
            self.lexical_vectorizer,
            os.path.join(self.models_dir, "lexical_vect.joblib"),
        )
        joblib.dump(
            self.community_vectorizer,
            os.path.join(self.models_dir, "community_vect.joblib"),
        )
        joblib.dump(self.imputer, os.path.join(self.models_dir, "meta_imputer.joblib"))
        joblib.dump(self.scaler, os.path.join(self.models_dir, "meta_scaler.joblib"))

        return {
            "lexical": X_train_lexical_sparse,
            "community": X_train_community_sparse,
            "meta": X_train_meta_dense,
            "semantic": X_train_semantic_dense,
        }

    def transform(self, df_test):
        """Transforms test data using fitted transformers."""
        print("Transforming test data...")

        # 1. Lexical
        text_test = self._get_text_data(df_test)
        X_test_lexical_sparse = self.lexical_vectorizer.transform(text_test)

        # 2. Community
        community_test = self._get_community_data(df_test)
        X_test_community_sparse = self.community_vectorizer.transform(community_test)

        # 3. Metadata
        meta_test = self._get_metadata_data(df_test)
        meta_test_imputed = self.imputer.transform(meta_test)
        X_test_meta_dense = self.scaler.transform(meta_test_imputed)

        # 4. Semantic
        self._load_embedding_model()
        X_test_semantic_dense = self.embedding_model.encode(
            text_test.tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return {
            "lexical": X_test_lexical_sparse,
            "community": X_test_community_sparse,
            "meta": X_test_meta_dense,
            "semantic": X_test_semantic_dense,
        }

    def _save_features(self, features, prefix):
        """Saves features to cache."""
        scipy.sparse.save_npz(
            os.path.join(self.cache_dir, f"{prefix}_lexical.npz"), features["lexical"]
        )
        scipy.sparse.save_npz(
            os.path.join(self.cache_dir, f"{prefix}_community.npz"),
            features["community"],
        )
        np.save(os.path.join(self.cache_dir, f"{prefix}_meta.npy"), features["meta"])
        np.save(
            os.path.join(self.cache_dir, f"{prefix}_semantic.npy"), features["semantic"]
        )

    def _load_features(self, prefix):
        """Loads features from cache."""
        try:
            lexical = scipy.sparse.load_npz(
                os.path.join(self.cache_dir, f"{prefix}_lexical.npz")
            )
            community = scipy.sparse.load_npz(
                os.path.join(self.cache_dir, f"{prefix}_community.npz")
            )
            meta = np.load(os.path.join(self.cache_dir, f"{prefix}_meta.npy"))
            semantic = np.load(os.path.join(self.cache_dir, f"{prefix}_semantic.npy"))
            return {
                "lexical": lexical,
                "community": community,
                "meta": meta,
                "semantic": semantic,
            }
        except FileNotFoundError:
            return None

    def execute(self, df_train, df_test, load_cached_data=True):
        """
        Main execution method. Checks cache, otherwise computes.
        Returns dictionary with 'train' and 'test' keys, each containing feature dicts.
        """
        if load_cached_data:
            train_feats = self._load_features("X_train")
            test_feats = self._load_features("X_test")
            if train_feats and test_feats:
                print("Loaded features from cache.")
                return {"train": train_feats, "test": test_feats}
            else:
                print("Cache incomplete or missing. Computing from scratch...")

        # Compute
        train_feats = self.fit_transform(df_train)
        test_feats = self.transform(df_test)

        # Save
        self._save_features(train_feats, "X_train")
        self._save_features(test_feats, "X_test")

        return {"train": train_feats, "test": test_feats}

    @staticmethod
    def get_lexical_input(features_dict):
        """
        Branch 1: Sparse Lexical
        Returns: Horizontal stack of [Text TF-IDF (Sparse) + Metadata (Dense)] -> Sparse
        """
        return scipy.sparse.hstack(
            [features_dict["lexical"], scipy.sparse.csr_matrix(features_dict["meta"])]
        ).tocsr()

    @staticmethod
    def get_community_input(features_dict):
        """
        Branch 2: Sparse Behavioral
        Returns: Horizontal stack of [Community TF-IDF (Sparse) + Metadata (Dense)] -> Sparse
        """
        return scipy.sparse.hstack(
            [features_dict["community"], scipy.sparse.csr_matrix(features_dict["meta"])]
        ).tocsr()

    @staticmethod
    def get_semantic_input(features_dict):
        """
        Branch 3: Dense Semantic
        Returns: Horizontal stack of [Semantic Embeddings (Dense) + Metadata (Dense)] -> Dense
        """
        return np.hstack([features_dict["semantic"], features_dict["meta"]])

    @staticmethod
    def get_metadata_input(features_dict):
        """
        Branch 4: Contextual
        Returns: Metadata (Dense)
        """
        return features_dict["meta"]
