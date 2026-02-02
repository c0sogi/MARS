import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import torch

from library.config import Config
from library.utils import save_artifact, load_artifact, set_seed


class FeaturePipeline:
    """
    Implements the hygienic feature extraction pipeline for the Granular Hept-View Ensemble.
    Generates four distinct views of the data:
    1. Lexical (Sparse TF-IDF of text)
    2. Behavioral (Sparse TF-IDF of subreddit history)
    3. Semantic (Dense Embeddings of text)
    4. Metadata (Contextual numerical features)
    """

    def __init__(self):
        self.cache_dir = Config.WORKING_DIR
        self.lexical_vectorizer = TfidfVectorizer(**Config.LEXICAL_VECTORIZER_PARAMS)
        self.community_vectorizer = TfidfVectorizer(
            **Config.COMMUNITY_VECTORIZER_PARAMS
        )
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")

        # Artifact names for caching
        self.artifacts = {
            "y_train": "y_train.npy",
            "X_meta_train": "X_train_meta.npy",
            "X_meta_test": "X_test_meta.npy",
            "X_lex_train": "X_train_lexical.npz",
            "X_lex_test": "X_test_lexical.npz",
            "X_beh_train": "X_train_community.npz",
            "X_beh_test": "X_test_community.npz",
            "X_sem_train": "X_train_semantic.npy",
            "X_sem_test": "X_test_semantic.npy",
        }

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        load_cached_data: bool = True,
    ) -> dict:
        """
        Main entry point. Checks cache, otherwise computes all features.

        Args:
            train_df: Union Training DataFrame (Train + Val).
            test_df: Test DataFrame.
            load_cached_data: Whether to attempt loading from disk.

        Returns:
            Dictionary containing all processed feature matrices and target.
        """
        set_seed(Config.RANDOM_SEED)

        # 1. Check Cache
        if load_cached_data and self._check_cache_exists():
            print("Loading features from cache...")
            return self._load_cache()

        print("Computing features from scratch...")

        # 2. Extract Target
        y_train = train_df[Config.TARGET_COL].values.astype(int)

        # 3. Process Views
        print("Processing Metadata (Contextual View)...")
        X_meta_train, X_meta_test = self._process_metadata(train_df, test_df)

        print("Processing Lexical Features (Text View)...")
        X_lex_train, X_lex_test = self._process_lexical(train_df, test_df)

        print("Processing Behavioral Features (History View)...")
        X_beh_train, X_beh_test = self._process_behavioral(train_df, test_df)

        print("Processing Semantic Features (Dense View)...")
        X_sem_train, X_sem_test = self._process_semantic(train_df, test_df)

        # 4. Construct Result
        data = {
            "y_train": y_train,
            "X_meta_train": X_meta_train,
            "X_meta_test": X_meta_test,
            "X_lex_train": X_lex_train,
            "X_lex_test": X_lex_test,
            "X_beh_train": X_beh_train,
            "X_beh_test": X_beh_test,
            "X_sem_train": X_sem_train,
            "X_sem_test": X_sem_test,
        }

        # 5. Save to Cache
        self._save_cache(data)

        return data

    def _process_metadata(self, train_df, test_df):
        """
        Extracts allow-listed numerical metadata, imputes missing values, and scales.
        """
        # Select columns
        cols = Config.METADATA_COLS

        # Extract raw values
        X_train = train_df[cols].copy()
        X_test = test_df[cols].copy()

        # Impute
        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train.astype(np.float32), X_test.astype(np.float32)

    def _process_lexical(self, train_df, test_df):
        """
        Sparse TF-IDF on concatenated Title + Body.
        Uses granular tokenization to preserve agency markers.
        """
        # Concatenate text columns
        train_text = self._concat_text(train_df)
        test_text = self._concat_text(test_df)

        # Vectorize
        X_train = self.lexical_vectorizer.fit_transform(train_text)
        X_test = self.lexical_vectorizer.transform(test_text)

        return X_train, X_test

    def _process_behavioral(self, train_df, test_df):
        """
        Sparse TF-IDF on Subreddit History (Bag-of-Concepts).
        """
        col = "requester_subreddits_at_request"

        # Join list of subreddits into space-separated strings
        # Handle cases where the column might be empty or not a list
        def join_subs(x):
            if isinstance(x, (list, np.ndarray)):
                return " ".join(x)
            return ""

        train_subs = train_df[col].apply(join_subs)
        test_subs = test_df[col].apply(join_subs)

        # Vectorize
        X_train = self.community_vectorizer.fit_transform(train_subs)
        X_test = self.community_vectorizer.transform(test_subs)

        return X_train, X_test

    def _process_semantic(self, train_df, test_df):
        """
        Dense Embeddings using Sentence Transformers.
        """
        # Concatenate text
        train_text = self._concat_text(train_df).tolist()
        test_text = self._concat_text(test_df).tolist()

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(Config.EMBEDDING_MODEL, device=device)

        # Encode
        # batch_size is adjustable, but defaults usually work fine
        X_train = model.encode(
            train_text, convert_to_numpy=True, show_progress_bar=False
        )
        X_test = model.encode(test_text, convert_to_numpy=True, show_progress_bar=False)

        return X_train, X_test

    def _concat_text(self, df):
        """Helper to concatenate title and body safely."""
        title = df[Config.TEXT_COLS[0]].fillna("").astype(str)
        body = df[Config.TEXT_COLS[1]].fillna("").astype(str)
        return title + " " + body

    def _check_cache_exists(self):
        """Verifies all required artifact files exist."""
        for filename in self.artifacts.values():
            path = os.path.join(self.cache_dir, filename)
            if not os.path.exists(path):
                return False
        return True

    def _load_cache(self):
        """Loads all artifacts from disk."""
        data = {}
        for key, filename in self.artifacts.items():
            path = os.path.join(self.cache_dir, filename)
            data[key] = load_artifact(path)
        return data

    def _save_cache(self, data):
        """Saves all artifacts to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        for key, filename in self.artifacts.items():
            path = os.path.join(self.cache_dir, filename)
            save_artifact(data[key], path)
