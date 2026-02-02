import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config
from library.utils import save_artifact, load_artifact
from library.data_factory import LABEL_MAP


def get_classical_features(train_df, test_df, load_cached_data=True):
    """
    Generates or loads classical features (TF-IDF and SVD) for the dataset.

    Args:
        train_df (pd.DataFrame): Training data containing 'text' and 'author'.
        test_df (pd.DataFrame): Test data containing 'text'.
        load_cached_data (bool): Whether to attempt loading from disk cache.

    Returns:
        tuple: (train_tfidf, test_tfidf, train_svd, test_svd, train_y)
            - train_tfidf, test_tfidf: Sparse matrices (scipy.sparse.csr_matrix)
            - train_svd, test_svd: Dense numpy arrays
            - train_y: Dense numpy array of encoded labels
    """
    # Define filenames for artifacts
    # Note: Sparse matrices use .npz, Dense use .npy
    tfidf_train_path = os.path.join(Config.WORKING_DIR, "tfidf_train.npz")
    tfidf_test_path = os.path.join(Config.WORKING_DIR, "tfidf_test.npz")
    svd_train_name = "svd_train.npy"
    svd_test_name = "svd_test.npy"
    labels_name = "train_labels.npy"

    # Check if all artifacts exist
    artifacts_exist = (
        os.path.exists(tfidf_train_path)
        and os.path.exists(tfidf_test_path)
        and os.path.exists(os.path.join(Config.WORKING_DIR, svd_train_name))
        and os.path.exists(os.path.join(Config.WORKING_DIR, svd_test_name))
        and os.path.exists(os.path.join(Config.WORKING_DIR, labels_name))
    )

    # 1. Try Loading from Cache
    if load_cached_data and artifacts_exist:
        try:
            print("Loading classical features from cache...")
            train_tfidf = scipy.sparse.load_npz(tfidf_train_path)
            test_tfidf = scipy.sparse.load_npz(tfidf_test_path)
            train_svd = load_artifact(svd_train_name)
            test_svd = load_artifact(svd_test_name)
            train_y = load_artifact(labels_name)
            return train_tfidf, test_tfidf, train_svd, test_svd, train_y
        except Exception as e:
            print(f"Failed to load cached features: {e}. Recomputing...")

    # 2. Compute from Scratch
    print("Computing classical features...")

    # Extract text and labels
    train_text = train_df["text"].astype(str).fillna("")
    test_text = test_df["text"].astype(str).fillna("")

    # Map labels to integers
    train_y = train_df["author"].map(LABEL_MAP).values.astype(np.int32)

    # --- A. TF-IDF (Sparse) ---
    print("Generating Word N-Grams...")
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=Config.NGRAM_RANGE_WORD,
        min_df=Config.MIN_DF,
        stop_words="english",
        sublinear_tf=True,
    )
    # Fit on train, transform both
    train_word = word_vectorizer.fit_transform(train_text)
    test_word = word_vectorizer.transform(test_text)

    print("Generating Character N-Grams...")
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=Config.NGRAM_RANGE_CHAR,
        min_df=Config.MIN_DF,
        sublinear_tf=True,
    )
    # Fit on train, transform both
    train_char = char_vectorizer.fit_transform(train_text)
    test_char = char_vectorizer.transform(test_text)

    print("Concatenating Sparse Features...")
    train_tfidf = scipy.sparse.hstack([train_word, train_char], format="csr")
    test_tfidf = scipy.sparse.hstack([test_word, test_char], format="csr")

    # --- B. SVD (Dense) ---
    print(f"Computing Truncated SVD ({Config.SVD_COMPONENTS} components)...")
    svd = TruncatedSVD(
        n_components=Config.SVD_COMPONENTS,
        algorithm="randomized",
        random_state=Config.SEED,
    )
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)

    # --- C. Save Artifacts ---
    print("Saving features to cache...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Save Sparse Matrices (using scipy directly as utils doesn't support sparse)
    scipy.sparse.save_npz(tfidf_train_path, train_tfidf)
    scipy.sparse.save_npz(tfidf_test_path, test_tfidf)

    # Save Dense Arrays (using utils)
    save_artifact(train_svd, svd_train_name)
    save_artifact(test_svd, svd_test_name)
    save_artifact(train_y, labels_name)

    return train_tfidf, test_tfidf, train_svd, test_svd, train_y
