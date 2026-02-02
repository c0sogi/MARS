import os
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import Config


class DualStreamVectorizer:
    """
    A custom vectorizer that combines word-level and character-level TF-IDF features.
    It wraps two sklearn TfidfVectorizers and concatenates their outputs.
    """

    def __init__(self):
        # Initialize vectorizers with parameters from Config
        self.word_vectorizer = TfidfVectorizer(**Config.WORD_TFIDF_PARAMS)
        self.char_vectorizer = TfidfVectorizer(**Config.CHAR_TFIDF_PARAMS)

    def fit(self, raw_documents):
        """
        Fits both the word and character vectorizers on the provided documents.
        """
        # Ensure input is string to handle any potential non-string types safely
        raw_documents = raw_documents.astype(str)

        self.word_vectorizer.fit(raw_documents)
        self.char_vectorizer.fit(raw_documents)
        return self

    def transform(self, raw_documents):
        """
        Transforms the documents using both vectorizers and concatenates the results.
        Returns a sparse CSR matrix.
        """
        raw_documents = raw_documents.astype(str)

        word_features = self.word_vectorizer.transform(raw_documents)
        char_features = self.char_vectorizer.transform(raw_documents)

        # Horizontally stack the sparse matrices
        combined_features = scipy.sparse.hstack(
            [word_features, char_features], format="csr"
        )
        return combined_features

    def fit_transform(self, raw_documents):
        """
        Fits and transforms the documents in a single step.
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)


def extract_features(X_train, X_val, X_test, load_cached_data=True):
    """
    Generates TF-IDF features for the training, validation, and test sets.
    Implements a caching mechanism to save/load sparse matrices to/from disk.

    Args:
        X_train (pd.Series): Training text data.
        X_val (pd.Series): Validation text data.
        X_test (pd.Series): Test text data.
        load_cached_data (bool): If True, attempts to load features from cache.

    Returns:
        tuple: A tuple containing (X_train_vec, X_val_vec, X_test_vec) as sparse matrices.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache filenames. Use a suffix for debug mode to avoid overwriting full data.
    suffix = "_debug" if Config.DEBUG else ""
    train_feat_path = os.path.join(Config.WORKING_DIR, f"train_features{suffix}.npz")
    val_feat_path = os.path.join(Config.WORKING_DIR, f"val_features{suffix}.npz")
    test_feat_path = os.path.join(Config.WORKING_DIR, f"test_features{suffix}.npz")

    # 1. Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_feat_path)
            and os.path.exists(val_feat_path)
            and os.path.exists(test_feat_path)
        ):
            print("Loading features from cache...")
            try:
                # Load sparse matrices
                X_train_vec = scipy.sparse.load_npz(train_feat_path)
                X_val_vec = scipy.sparse.load_npz(val_feat_path)
                X_test_vec = scipy.sparse.load_npz(test_feat_path)
                return X_train_vec, X_val_vec, X_test_vec
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing features...")

    # 2. Compute features if cache is missing or load failed
    print("Extracting features (TF-IDF)...")

    vectorizer = DualStreamVectorizer()

    # Fit on training data and transform
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transform validation and test data
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    # 3. Save to cache
    print(f"Saving features to {Config.WORKING_DIR}...")
    try:
        scipy.sparse.save_npz(train_feat_path, X_train_vec)
        scipy.sparse.save_npz(val_feat_path, X_val_vec)
        scipy.sparse.save_npz(test_feat_path, X_test_vec)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return X_train_vec, X_val_vec, X_test_vec
