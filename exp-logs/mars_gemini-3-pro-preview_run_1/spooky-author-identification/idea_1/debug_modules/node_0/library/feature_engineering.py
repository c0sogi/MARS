import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from library.config import Config


class HybridTfidfVectorizer:
    """
    A wrapper class that combines word-level and character-level TF-IDF features.
    """

    def __init__(self, word_params, char_params):
        """
        Initialize with parameters for both vectorizers.
        """
        self.word_vectorizer = TfidfVectorizer(**word_params)
        self.char_vectorizer = TfidfVectorizer(**char_params)

    def fit(self, raw_documents):
        """
        Fit both vectorizers on the raw documents.
        """
        self.word_vectorizer.fit(raw_documents)
        self.char_vectorizer.fit(raw_documents)
        return self

    def transform(self, raw_documents):
        """
        Transform documents into a stacked sparse matrix of word and char features.
        """
        word_features = self.word_vectorizer.transform(raw_documents)
        char_features = self.char_vectorizer.transform(raw_documents)
        return hstack([word_features, char_features])


def extract_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Extracts features from the provided dataframes. Handles caching of the
    generated feature matrices and labels to disk using .npy format.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, classes)
            X arrays are dense numpy arrays (float32).
            y arrays are 1D numpy arrays of encoded integers.
            classes is a numpy array of the original class strings.
    """
    # Determine cache directory and filenames
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Append suffix for debug mode to avoid cache collisions
    suffix = "_debug" if Config.DEBUG else ""

    filenames = {
        "X_train": f"X_train{suffix}.npy",
        "y_train": f"y_train{suffix}.npy",
        "X_val": f"X_val{suffix}.npy",
        "y_val": f"y_val{suffix}.npy",
        "X_test": f"X_test{suffix}.npy",
        "classes": f"classes{suffix}.npy",
    }

    paths = {k: os.path.join(cache_dir, v) for k, v in filenames.items()}

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and cache_exists:
        print("Loading features from cache...")
        X_train = np.load(paths["X_train"])
        y_train = np.load(paths["y_train"])
        X_val = np.load(paths["X_val"])
        y_val = np.load(paths["y_val"])
        X_test = np.load(paths["X_test"])
        classes = np.load(paths["classes"], allow_pickle=True)
        return X_train, y_train, X_val, y_val, X_test, classes

    print("Computing features from scratch...")

    # Initialize the hybrid vectorizer
    vectorizer = HybridTfidfVectorizer(
        Config.WORD_TFIDF_PARAMS, Config.CHAR_TFIDF_PARAMS
    )

    # Fit on training text
    vectorizer.fit(train_df["text"])

    # Transform all datasets
    # Result is scipy.sparse.csr_matrix
    X_train_sparse = vectorizer.transform(train_df["text"])
    X_val_sparse = vectorizer.transform(val_df["text"])
    X_test_sparse = vectorizer.transform(test_df["text"])

    # Encode target labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["author"])
    y_val = le.transform(val_df["author"])
    classes = le.classes_

    # Convert sparse matrices to dense numpy arrays for .npy storage
    # Using float32 to optimize memory usage (approx 4GB for full dataset)
    X_train = X_train_sparse.toarray().astype(np.float32)
    X_val = X_val_sparse.toarray().astype(np.float32)
    X_test = X_test_sparse.toarray().astype(np.float32)

    # Save to cache
    print(f"Saving features to {cache_dir}...")
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["X_test"], X_test)
    np.save(paths["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, classes
