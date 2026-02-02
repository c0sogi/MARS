import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import torch
import joblib

from library import config
from library import utils


class FeaturePipeline:
    def __init__(self, load_cached_data: bool = True):
        self.load_cached_data = load_cached_data
        self.logger = utils.get_logger(__name__)
        self.cache_dir = config.WORKING_DIR  # Use the specific idea directory

        # Define cache file paths
        self.files = {
            "y_train": "y_train.npy",
            "y_val": "y_val.npy",
            "X_train_meta": "X_train_metadata.npy",
            "X_val_meta": "X_val_metadata.npy",
            "X_test_meta": "X_test_metadata.npy",
            "X_train_lexical": "X_train_lexical.npz",
            "X_val_lexical": "X_val_lexical.npz",
            "X_test_lexical": "X_test_lexical.npz",
            "X_train_behavioral": "X_train_behavioral.npz",
            "X_val_behavioral": "X_val_behavioral.npz",
            "X_test_behavioral": "X_test_behavioral.npz",
            "X_train_semantic": "X_train_semantic.npy",
            "X_val_semantic": "X_val_semantic.npy",
            "X_test_semantic": "X_test_semantic.npy",
        }

    def execute(self):
        """
        Main execution method. Checks cache, loads if available, else processes data.
        Returns a dictionary containing all data views.
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        if self.load_cached_data and self._check_cache():
            self.logger.info("Loading features from cache...")
            return self._load_cache()

        self.logger.info("Cache missing or reload requested. Processing features...")
        data = self._process_data()
        self._save_cache(data)
        return data

    def _check_cache(self):
        for fname in self.files.values():
            if not os.path.exists(os.path.join(self.cache_dir, fname)):
                return False
        return True

    def _load_cache(self):
        data = {}
        for key, fname in self.files.items():
            path = os.path.join(self.cache_dir, fname)
            if fname.endswith(".npy"):
                data[key] = np.load(path)
            elif fname.endswith(".npz"):
                data[key] = sparse.load_npz(path)
        return data

    def _save_cache(self, data):
        self.logger.info("Saving features to cache...")
        for key, fname in self.files.items():
            path = os.path.join(self.cache_dir, fname)
            if fname.endswith(".npy"):
                np.save(path, data[key])
            elif fname.endswith(".npz"):
                sparse.save_npz(path, data[key])

    def _load_raw_data(self):
        self.logger.info("Loading raw parquet files...")
        train_df = pd.read_parquet(config.TRAIN_PATH)
        val_df = pd.read_parquet(config.VAL_PATH)
        test_df = pd.read_parquet(config.TEST_PATH)
        return train_df, val_df, test_df

    def _process_metadata(self, train_df, val_df, test_df):
        self.logger.info("Processing Metadata View...")

        # Select columns
        cols = config.METADATA_COLS

        # Extract arrays
        X_train = train_df[cols].values
        X_val = val_df[cols].values
        X_test = test_df[cols].values

        # Impute
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return (
            X_train.astype(np.float32),
            X_val.astype(np.float32),
            X_test.astype(np.float32),
        )

    def _process_lexical(self, train_df, val_df, test_df):
        self.logger.info("Processing Lexical View (TF-IDF)...")

        # Concatenate text columns
        def concat_text(df):
            # Fill NaNs with empty string and join
            return df[config.TEXT_COLS].fillna("").astype(str).agg(" ".join, axis=1)

        train_text = concat_text(train_df)
        val_text = concat_text(val_df)
        test_text = concat_text(test_df)

        # Vectorize
        vectorizer = TfidfVectorizer(**config.TFIDF_PARAMS)
        X_train = vectorizer.fit_transform(train_text)
        X_val = vectorizer.transform(val_text)
        X_test = vectorizer.transform(test_text)

        return X_train, X_val, X_test, train_text, val_text, test_text

    def _process_behavioral(self, train_df, val_df, test_df):
        self.logger.info("Processing Behavioral View (Subreddits)...")

        col = config.SUBREDDIT_COL

        def process_subreddits(df):
            # Convert list to space-separated string
            # Handle cases where it might be None or empty list
            return df[col].apply(
                lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else ""
            )

        train_subs = process_subreddits(train_df)
        val_subs = process_subreddits(val_df)
        test_subs = process_subreddits(test_df)

        # Vectorize
        vectorizer = TfidfVectorizer(**config.SUBREDDIT_TFIDF_PARAMS)
        X_train = vectorizer.fit_transform(train_subs)
        X_val = vectorizer.transform(val_subs)
        X_test = vectorizer.transform(test_subs)

        return X_train, X_val, X_test

    def _process_semantic(self, train_text, val_text, test_text):
        self.logger.info("Processing Semantic View (Embeddings)...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {device}")

        model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

        # Encode
        # Note: Converting Series to list for sentence-transformers
        X_train = model.encode(
            train_text.tolist(), show_progress_bar=False, convert_to_numpy=True
        )
        X_val = model.encode(
            val_text.tolist(), show_progress_bar=False, convert_to_numpy=True
        )
        X_test = model.encode(
            test_text.tolist(), show_progress_bar=False, convert_to_numpy=True
        )

        # Scale embeddings (StandardScaler on dense embeddings is often beneficial for gradient boosters)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return (
            X_train.astype(np.float32),
            X_val.astype(np.float32),
            X_test.astype(np.float32),
        )

    def _process_data(self):
        train_df, val_df, test_df = self._load_raw_data()

        # Targets
        y_train = train_df[config.TARGET_COL].values.astype(int)
        y_val = val_df[config.TARGET_COL].values.astype(int)

        # 1. Metadata View
        X_train_meta, X_val_meta, X_test_meta = self._process_metadata(
            train_df, val_df, test_df
        )

        # 2. Lexical View (and get raw text for semantics)
        X_train_lex, X_val_lex, X_test_lex, train_text, val_text, test_text = (
            self._process_lexical(train_df, val_df, test_df)
        )

        # 3. Behavioral View
        X_train_beh, X_val_beh, X_test_beh = self._process_behavioral(
            train_df, val_df, test_df
        )

        # 4. Semantic View
        X_train_sem, X_val_sem, X_test_sem = self._process_semantic(
            train_text, val_text, test_text
        )

        return {
            "y_train": y_train,
            "y_val": y_val,
            "X_train_meta": X_train_meta,
            "X_val_meta": X_val_meta,
            "X_test_meta": X_test_meta,
            "X_train_lexical": X_train_lex,
            "X_val_lexical": X_val_lex,
            "X_test_lexical": X_test_lex,
            "X_train_behavioral": X_train_beh,
            "X_val_behavioral": X_val_beh,
            "X_test_behavioral": X_test_beh,
            "X_train_semantic": X_train_sem,
            "X_val_semantic": X_val_sem,
            "X_test_semantic": X_test_sem,
        }
