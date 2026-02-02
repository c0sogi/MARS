import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import Config
from library.dataset import load_data


def save_sparse_csr(filename_prefix, matrix):
    """
    Saves a scipy.sparse.csr_matrix to multiple .npy files.
    Args:
        filename_prefix (str): Base path and prefix for the files.
        matrix (scipy.sparse.csr_matrix): The matrix to save.
    """
    np.save(f"{filename_prefix}_data.npy", matrix.data)
    np.save(f"{filename_prefix}_indices.npy", matrix.indices)
    np.save(f"{filename_prefix}_indptr.npy", matrix.indptr)
    np.save(f"{filename_prefix}_shape.npy", np.array(matrix.shape))


def load_sparse_csr(filename_prefix):
    """
    Loads a scipy.sparse.csr_matrix from multiple .npy files.
    Args:
        filename_prefix (str): Base path and prefix for the files.
    Returns:
        scipy.sparse.csr_matrix: The loaded matrix.
    """
    data = np.load(f"{filename_prefix}_data.npy")
    indices = np.load(f"{filename_prefix}_indices.npy")
    indptr = np.load(f"{filename_prefix}_indptr.npy")
    shape = np.load(f"{filename_prefix}_shape.npy")
    return sp.csr_matrix((data, indices, indptr), shape=tuple(shape))


def get_tfidf_features(load_cached_data=True):
    """
    Generates or loads TF-IDF features for Train, Validation, and Test sets.

    Implements a pipeline combining Word N-grams and Character N-grams.
    Fits on the combined corpus (Train + Val + Test) to build a comprehensive vocabulary.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed features from disk.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test)
            X_* are scipy.sparse.csr_matrix
            y_* are numpy.ndarray
    """
    # Define cache directory and paths
    feature_dir = os.path.join(Config.WORKING_DIR, "tfidf_features")
    os.makedirs(feature_dir, exist_ok=True)

    prefixes = {
        "train": os.path.join(feature_dir, "X_train_tfidf"),
        "val": os.path.join(feature_dir, "X_val_tfidf"),
        "test": os.path.join(feature_dir, "X_test_tfidf"),
        "y_train": os.path.join(feature_dir, "y_train"),
        "y_val": os.path.join(feature_dir, "y_val"),
    }

    # Check if all cache files exist
    cache_exists = True
    # Check sparse matrix files
    for split in ["train", "val", "test"]:
        for suffix in ["_data.npy", "_indices.npy", "_indptr.npy", "_shape.npy"]:
            if not os.path.exists(prefixes[split] + suffix):
                cache_exists = False
                break
    # Check label files
    if not os.path.exists(prefixes["y_train"] + ".npy") or not os.path.exists(
        prefixes["y_val"] + ".npy"
    ):
        cache_exists = False

    if load_cached_data and cache_exists:
        # print("Loading TF-IDF features from cache...")
        X_train = load_sparse_csr(prefixes["train"])
        X_val = load_sparse_csr(prefixes["val"])
        X_test = load_sparse_csr(prefixes["test"])
        y_train = np.load(prefixes["y_train"] + ".npy")
        y_val = np.load(prefixes["y_val"] + ".npy")
        return X_train, y_train, X_val, y_val, X_test

    # print("Computing TF-IDF features...")

    # 1. Load Raw Data
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    # Extract text and labels
    train_text = train_df[Config.TEXT_COL].fillna("").astype(str).values
    val_text = val_df[Config.TEXT_COL].fillna("").astype(str).values
    test_text = test_df[Config.TEXT_COL].fillna("").astype(str).values

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # 2. Initialize Vectorizers
    # Word-level vectorizer
    word_vectorizer = TfidfVectorizer(
        ngram_range=Config.TFIDF_NGRAM_RANGE_WORD,
        max_features=Config.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        analyzer="word",
        token_pattern=r"\w{1,}",
    )

    # Character-level vectorizer
    char_vectorizer = TfidfVectorizer(
        ngram_range=Config.TFIDF_NGRAM_RANGE_CHAR,
        max_features=Config.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        analyzer="char",
    )

    # 3. Fit on Combined Corpus (Train + Val + Test)
    # This ensures the vocabulary covers all seen text (transductive setting for features)
    all_text = np.concatenate([train_text, val_text, test_text])

    # print("Fitting Word Vectorizer...")
    word_vectorizer.fit(all_text)
    # print("Fitting Char Vectorizer...")
    char_vectorizer.fit(all_text)

    # 4. Transform Splits
    # print("Transforming Train...")
    X_train_word = word_vectorizer.transform(train_text)
    X_train_char = char_vectorizer.transform(train_text)
    X_train = sp.hstack([X_train_word, X_train_char], format="csr")

    # print("Transforming Val...")
    X_val_word = word_vectorizer.transform(val_text)
    X_val_char = char_vectorizer.transform(val_text)
    X_val = sp.hstack([X_val_word, X_val_char], format="csr")

    # print("Transforming Test...")
    X_test_word = word_vectorizer.transform(test_text)
    X_test_char = char_vectorizer.transform(test_text)
    X_test = sp.hstack([X_test_word, X_test_char], format="csr")

    # 5. Save to Cache
    # print("Saving features to cache...")
    save_sparse_csr(prefixes["train"], X_train)
    save_sparse_csr(prefixes["val"], X_val)
    save_sparse_csr(prefixes["test"], X_test)
    np.save(prefixes["y_train"] + ".npy", y_train)
    np.save(prefixes["y_val"] + ".npy", y_val)

    return X_train, y_train, X_val, y_val, X_test
