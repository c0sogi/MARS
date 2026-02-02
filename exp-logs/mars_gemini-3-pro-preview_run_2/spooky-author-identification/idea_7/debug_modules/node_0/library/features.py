import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from library.config import Config
from library.utils import save_artifact, load_artifact, seed_everything


class FeatureEngineer:
    """
    Handles feature engineering for the author identification task.
    Generates:
    1. Sparse TF-IDF features (Word + Char n-grams) for Linear Models.
    2. Dense features (SVD of TF-IDF + Stylometric features) for XGBoost.
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.stop_words = set(ENGLISH_STOP_WORDS)

        # Initialize Vectorizers
        self.word_vectorizer = TfidfVectorizer(
            min_df=Config.TFIDF_MIN_DF,
            ngram_range=Config.TFIDF_WORD_NGRAM_RANGE,
            analyzer="word",
            stop_words="english",
        )
        self.char_vectorizer = TfidfVectorizer(
            min_df=Config.TFIDF_MIN_DF,
            ngram_range=Config.TFIDF_CHAR_NGRAM_RANGE,
            analyzer="char",
        )

        # Initialize SVD
        self.svd = TruncatedSVD(
            n_components=Config.SVD_N_COMPONENTS, random_state=Config.SEED
        )

        # Initialize Label Encoder
        self.le = LabelEncoder()

    def extract_stylometric_features(self, text_series):
        """
        Extracts explicit stylometric features from text.
        Features:
        - Character count
        - Word count
        - Average word length
        - Punctuation density
        - Stopword count
        """
        # Convert to string to ensure safety
        texts = text_series.astype(str).tolist()

        features = []

        for text in texts:
            row_feats = []

            # 1. Length Metrics
            char_len = len(text)
            words = text.split()
            word_len = len(words)
            avg_word_len = np.mean([len(w) for w in words]) if word_len > 0 else 0

            row_feats.extend([char_len, word_len, avg_word_len])

            # 2. Punctuation Counts
            for p_char in Config.PUNCTUATION_CHARS:
                row_feats.append(text.count(p_char))

            # 3. Stopword Usage
            stopword_count = sum(1 for w in words if w.lower() in self.stop_words)
            row_feats.append(stopword_count)

            features.append(row_feats)

        return np.array(features, dtype=np.float32)

    def get_tfidf_features(self, train_text, test_text):
        """
        Generates concatenated Word and Character TF-IDF sparse matrices.
        Fits on train_text, transforms both.
        """
        # Word N-grams
        train_word = self.word_vectorizer.fit_transform(train_text)
        test_word = self.word_vectorizer.transform(test_text)

        # Char N-grams
        train_char = self.char_vectorizer.fit_transform(train_text)
        test_char = self.char_vectorizer.transform(test_text)

        # Concatenate
        train_tfidf = scipy.sparse.hstack([train_word, train_char])
        test_tfidf = scipy.sparse.hstack([test_word, test_char])

        return train_tfidf, test_tfidf

    def apply_svd(self, train_tfidf, test_tfidf):
        """
        Applies Truncated SVD to sparse TF-IDF matrices.
        """
        train_svd = self.svd.fit_transform(train_tfidf)
        test_svd = self.svd.transform(test_tfidf)

        return train_svd, test_svd

    def process_data(self, load_cached_data=True):
        """
        Main pipeline execution.
        1. Checks cache.
        2. If not cached, loads metadata, computes features.
        3. Saves to cache.
        4. Returns dictionary of data.
        """
        # Define paths for sparse matrices (handled locally as utils doesn't support npz)
        sparse_train_path = os.path.join(
            Config.WORKING_DIR, "train_features_sparse.npz"
        )
        sparse_test_path = os.path.join(Config.WORKING_DIR, "test_features_sparse.npz")

        # Check if all artifacts exist
        artifacts_exist = (
            os.path.exists(Config.CACHE_TRAIN_FEATURES)
            and os.path.exists(Config.CACHE_TEST_FEATURES)
            and os.path.exists(Config.CACHE_TRAIN_LABELS)
            and os.path.exists(Config.CACHE_LABEL_ENCODER)
            and os.path.exists(sparse_train_path)
            and os.path.exists(sparse_test_path)
        )

        if load_cached_data and artifacts_exist:
            print("Loading features from cache...")
            train_dense = load_artifact(Config.CACHE_TRAIN_FEATURES)
            test_dense = load_artifact(Config.CACHE_TEST_FEATURES)
            y_train = load_artifact(Config.CACHE_TRAIN_LABELS)

            # Load Label Encoder classes
            le_classes = load_artifact(Config.CACHE_LABEL_ENCODER)
            self.le.classes_ = le_classes

            # Load Sparse
            train_sparse = scipy.sparse.load_npz(sparse_train_path)
            test_sparse = scipy.sparse.load_npz(sparse_test_path)

            # We also need the raw dataframes to return IDs
            # Re-loading metadata is fast
            df_train = pd.read_csv(Config.TRAIN_META_PATH)
            df_val = pd.read_csv(Config.VAL_META_PATH)
            df_train_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
            df_test = pd.read_csv(Config.TEST_META_PATH)

            return {
                "train_df": df_train_full,
                "test_df": df_test,
                "X_train_sparse": train_sparse,
                "X_test_sparse": test_sparse,
                "X_train_dense": train_dense,
                "X_test_dense": test_dense,
                "y_train": y_train,
                "label_encoder": self.le,
            }

        print("Computing features from scratch...")

        # 1. Load Metadata
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        df_val = pd.read_csv(Config.VAL_META_PATH)
        # Combine Train and Val for full training set (CV handled later)
        df_train_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
        df_test = pd.read_csv(Config.TEST_META_PATH)

        # 2. Encode Labels
        y_train = self.le.fit_transform(df_train_full["author"])

        # 3. Text Preprocessing (Basic safety)
        train_text = df_train_full["text"].fillna("").astype(str)
        test_text = df_test["text"].fillna("").astype(str)

        # 4. Stylometric Features (Dense)
        print("Extracting Stylometric features...")
        train_stylo = self.extract_stylometric_features(train_text)
        test_stylo = self.extract_stylometric_features(test_text)

        # 5. TF-IDF Features (Sparse)
        print("Extracting TF-IDF features...")
        train_sparse, test_sparse = self.get_tfidf_features(train_text, test_text)

        # 6. SVD Features (Dense)
        print("Applying SVD...")
        train_svd, test_svd = self.apply_svd(train_sparse, test_sparse)

        # 7. Combine Dense Features (SVD + Stylo)
        train_dense = np.hstack([train_svd, train_stylo])
        test_dense = np.hstack([test_svd, test_stylo])

        # 8. Save Artifacts
        print("Saving artifacts to cache...")
        save_artifact(train_dense, Config.CACHE_TRAIN_FEATURES)
        save_artifact(test_dense, Config.CACHE_TEST_FEATURES)
        save_artifact(y_train, Config.CACHE_TRAIN_LABELS)
        save_artifact(self.le.classes_, Config.CACHE_LABEL_ENCODER)

        # Save sparse matrices
        scipy.sparse.save_npz(sparse_train_path, train_sparse)
        scipy.sparse.save_npz(sparse_test_path, test_sparse)

        return {
            "train_df": df_train_full,
            "test_df": df_test,
            "X_train_sparse": train_sparse,
            "X_test_sparse": test_sparse,
            "X_train_dense": train_dense,
            "X_test_dense": test_dense,
            "y_train": y_train,
            "label_encoder": self.le,
        }
