import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import library.config as config


class FeatureExtractor:
    """
    Encapsulates the feature extraction logic using TF-IDF vectorizers.
    Generates both word-level and character-level n-gram features.
    """

    def __init__(
        self,
        word_ngram_range=config.WORD_NGRAM_RANGE,
        char_ngram_range=config.CHAR_NGRAM_RANGE,
        max_features_word=config.MAX_FEATURES_WORD,
        max_features_char=config.MAX_FEATURES_CHAR,
    ):

        # Word n-gram vectorizer
        # Using a token pattern that includes single-letter words which can be indicative of style
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=word_ngram_range,
            max_features=max_features_word,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )

        # Character n-gram vectorizer
        self.char_vectorizer = TfidfVectorizer(
            ngram_range=char_ngram_range,
            max_features=max_features_char,
            analyzer="char",
        )

    def fit_transform(self, text_data):
        """
        Fits the vectorizers on the provided text data and returns the concatenated feature matrix.
        """
        # Fit and transform word n-grams
        word_features = self.word_vectorizer.fit_transform(text_data)

        # Fit and transform char n-grams
        char_features = self.char_vectorizer.fit_transform(text_data)

        # Concatenate horizontally
        return scipy.sparse.hstack([word_features, char_features], format="csr")

    def transform(self, text_data):
        """
        Transforms the provided text data using the already fitted vectorizers.
        """
        word_features = self.word_vectorizer.transform(text_data)
        char_features = self.char_vectorizer.transform(text_data)

        return scipy.sparse.hstack([word_features, char_features], format="csr")


def save_sparse_matrix(filename_prefix, matrix):
    """
    Saves a CSR matrix to multiple .npy files (data, indices, indptr, shape).
    This avoids using pickle.
    """
    # Ensure matrix is CSR
    if not scipy.sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()

    np.save(f"{filename_prefix}_data.npy", matrix.data)
    np.save(f"{filename_prefix}_indices.npy", matrix.indices)
    np.save(f"{filename_prefix}_indptr.npy", matrix.indptr)
    np.save(f"{filename_prefix}_shape.npy", np.array(matrix.shape))


def load_sparse_matrix(filename_prefix):
    """
    Loads a CSR matrix from multiple .npy files.
    """
    data = np.load(f"{filename_prefix}_data.npy")
    indices = np.load(f"{filename_prefix}_indices.npy")
    indptr = np.load(f"{filename_prefix}_indptr.npy")
    shape = np.load(f"{filename_prefix}_shape.npy")

    return scipy.sparse.csr_matrix((data, indices, indptr), shape=tuple(shape))


def check_cache_exists(prefix):
    """
    Checks if all component files for a sparse matrix exist.
    """
    return (
        os.path.exists(f"{prefix}_data.npy")
        and os.path.exists(f"{prefix}_indices.npy")
        and os.path.exists(f"{prefix}_indptr.npy")
        and os.path.exists(f"{prefix}_shape.npy")
    )


def extract_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Main function to extract features from DataFrames.
    Implements caching to disk using .npy files.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test)
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Define file path prefixes
    train_x_prefix = os.path.join(config.CACHE_DIR, "X_train")
    val_x_prefix = os.path.join(config.CACHE_DIR, "X_val")
    test_x_prefix = os.path.join(config.CACHE_DIR, "X_test")

    train_y_path = os.path.join(config.CACHE_DIR, "y_train.npy")
    val_y_path = os.path.join(config.CACHE_DIR, "y_val.npy")

    # Check if cache is valid
    cache_valid = (
        check_cache_exists(train_x_prefix)
        and check_cache_exists(val_x_prefix)
        and check_cache_exists(test_x_prefix)
        and os.path.exists(train_y_path)
        and os.path.exists(val_y_path)
    )

    # 1. Try to load from cache
    if load_cached_data and cache_valid:
        print("Loading features from cache...")
        try:
            X_train = load_sparse_matrix(train_x_prefix)
            X_val = load_sparse_matrix(val_x_prefix)
            X_test = load_sparse_matrix(test_x_prefix)
            y_train = np.load(train_y_path)
            y_val = np.load(val_y_path)

            return X_train, y_train, X_val, y_val, X_test
        except Exception as e:
            print(f"Error loading cache: {e}. Processing from scratch...")

    # 2. Process from scratch
    print("Extracting features from text...")

    # Prepare text inputs (ensure string format)
    train_text = train_df["text"].fillna("").astype(str)
    val_text = val_df["text"].fillna("").astype(str)
    test_text = test_df["text"].fillna("").astype(str)

    # Initialize FeatureExtractor
    extractor = FeatureExtractor()

    # Fit on Train, Transform all
    print("Vectorizing training data...")
    X_train = extractor.fit_transform(train_text)

    print("Vectorizing validation and test data...")
    X_val = extractor.transform(val_text)
    X_test = extractor.transform(test_text)

    # Encode Labels
    print("Encoding labels...")
    le = LabelEncoder()
    # Fit on the fixed classes from config to ensure consistent mapping (EAP=0, HPL=1, MWS=2)
    le.fit(config.CLASSES)

    if "author" in train_df.columns:
        y_train = le.transform(train_df["author"])
    else:
        raise ValueError("Training dataframe is missing 'author' column.")

    if "author" in val_df.columns:
        y_val = le.transform(val_df["author"])
    else:
        raise ValueError("Validation dataframe is missing 'author' column.")

    # 3. Save to cache
    print(f"Saving features to cache at {config.CACHE_DIR}...")
    save_sparse_matrix(train_x_prefix, X_train)
    save_sparse_matrix(val_x_prefix, X_val)
    save_sparse_matrix(test_x_prefix, X_test)
    np.save(train_y_path, y_train)
    np.save(val_y_path, y_val)

    return X_train, y_train, X_val, y_val, X_test
