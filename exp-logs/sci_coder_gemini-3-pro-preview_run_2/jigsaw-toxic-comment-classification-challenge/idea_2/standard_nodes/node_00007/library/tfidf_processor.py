import os
import numpy as np
import scipy.sparse
from scipy.sparse import hstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import Config


class TfidfVectorizationPipeline:
    """
    Manages the extraction and caching of TF-IDF features (Word + Char n-grams)
    for the linear model branch of the ensemble.
    """

    def __init__(self):
        self.word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            stop_words="english",
            ngram_range=Config.TFIDF_WORD_NGRAM_RANGE,
            max_features=Config.TFIDF_WORD_MAX_FEATURES,
        )

        self.char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="char",
            ngram_range=Config.TFIDF_CHAR_NGRAM_RANGE,
            max_features=Config.TFIDF_CHAR_MAX_FEATURES,
        )

    def run(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Runs the vectorization pipeline. Checks for cached sparse matrices first.
        If not found or forced reload, computes features and saves them.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            test_df (pd.DataFrame): Test data.
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            tuple: (X_train, X_val, X_test) as scipy.sparse.csr_matrix
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Define cache filenames
        # We append _debug if running in debug mode to avoid polluting the full cache
        suffix = "_debug" if Config.DEBUG else ""
        train_cache_path = os.path.join(Config.WORKING_DIR, f"tfidf_train{suffix}.npz")
        val_cache_path = os.path.join(Config.WORKING_DIR, f"tfidf_val{suffix}.npz")
        test_cache_path = os.path.join(Config.WORKING_DIR, f"tfidf_test{suffix}.npz")

        # 1. Attempt to load from cache
        if load_cached_data:
            if (
                os.path.exists(train_cache_path)
                and os.path.exists(val_cache_path)
                and os.path.exists(test_cache_path)
            ):
                print(f"Loading TF-IDF features from cache: {Config.WORKING_DIR} ...")
                try:
                    X_train = load_npz(train_cache_path)
                    X_val = load_npz(val_cache_path)
                    X_test = load_npz(test_cache_path)
                    print(
                        f"Loaded features shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}"
                    )
                    return X_train, X_val, X_test
                except Exception as e:
                    print(f"Error loading cache: {e}. Recomputing features...")
            else:
                print("Cache miss for TF-IDF features. Computing from scratch...")
        else:
            print("Force reload requested. Computing TF-IDF features from scratch...")

        # 2. Compute Features
        print("Extracting text data...")
        train_text = train_df[Config.TEXT_COL]
        val_text = val_df[Config.TEXT_COL]
        test_text = test_df[Config.TEXT_COL]

        # A. Word Vectorization
        print(f"Fitting Word TF-IDF (max_features={Config.TFIDF_WORD_MAX_FEATURES})...")
        self.word_vectorizer.fit(train_text)

        print("Transforming Word features...")
        train_word = self.word_vectorizer.transform(train_text)
        val_word = self.word_vectorizer.transform(val_text)
        test_word = self.word_vectorizer.transform(test_text)

        # B. Char Vectorization
        print(f"Fitting Char TF-IDF (max_features={Config.TFIDF_CHAR_MAX_FEATURES})...")
        self.char_vectorizer.fit(train_text)

        print("Transforming Char features...")
        train_char = self.char_vectorizer.transform(train_text)
        val_char = self.char_vectorizer.transform(val_text)
        test_char = self.char_vectorizer.transform(test_text)

        # C. Stack Features
        print("Stacking Word and Char features...")
        X_train = hstack([train_word, train_char]).tocsr()
        X_val = hstack([val_word, val_char]).tocsr()
        X_test = hstack([test_word, test_char]).tocsr()

        print(f"Feature extraction complete. Final Shape: {X_train.shape}")

        # 3. Save to Cache
        print(f"Saving TF-IDF features to {Config.WORKING_DIR}...")
        try:
            save_npz(train_cache_path, X_train)
            save_npz(val_cache_path, X_val)
            save_npz(test_cache_path, X_test)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

        return X_train, X_val, X_test
