import os
import string
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


def _compute_meta_features_dataframe(df):
    """
    Internal helper to compute explicit meta-features for a given DataFrame.

    Features computed:
    1. Sentence Character Length
    2. Word Count
    3. Punctuation Density

    Args:
        df (pd.DataFrame): DataFrame containing a 'text' column.

    Returns:
        np.ndarray: A dense numpy array of shape (n_samples, 3).
    """
    # Ensure text is string and handle NaNs
    texts = df["text"].fillna("").astype(str)

    # 1. Character Length
    char_len = texts.apply(len).values

    # 2. Word Count
    # Splitting by whitespace
    word_count = texts.apply(lambda x: len(x.split())).values

    # 3. Punctuation Density
    # Pre-compute punctuation set for speed
    punct_chars = set(string.punctuation)

    # Count punctuation characters in each text
    punct_count = texts.apply(lambda x: sum(1 for c in x if c in punct_chars)).values

    # Calculate density: count / length
    # Add epsilon or max(len, 1) to avoid division by zero
    safe_char_len = np.maximum(char_len, 1)
    punct_density = punct_count / safe_char_len

    # Stack features into a single matrix
    # Shape: (N, 3)
    features = np.column_stack((char_len, word_count, punct_density))

    return features.astype(np.float32)


def get_meta_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates or loads explicit meta-features for the Meta-Learner.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_meta, val_meta, test_meta) as numpy arrays.
    """
    # Define cache paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_path = os.path.join(cache_dir, "meta_features_train.npy")
    val_path = os.path.join(cache_dir, "meta_features_val.npy")
    test_path = os.path.join(cache_dir, "meta_features_test.npy")

    files_exist = all(os.path.exists(p) for p in [train_path, val_path, test_path])

    # 1. Try to load from cache
    if load_cached_data and files_exist:
        print("Loading meta-features from cache...")
        try:
            train_meta = np.load(train_path)
            val_meta = np.load(val_path)
            test_meta = np.load(test_path)
            return train_meta, val_meta, test_meta
        except Exception as e:
            print(f"Error loading meta-feature cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing meta-features...")
    train_meta = _compute_meta_features_dataframe(train_df)
    val_meta = _compute_meta_features_dataframe(val_df)
    test_meta = _compute_meta_features_dataframe(test_df)

    # 3. Save to cache
    print(f"Saving meta-features to {cache_dir}...")
    np.save(train_path, train_meta)
    np.save(val_path, val_meta)
    np.save(test_path, test_meta)

    return train_meta, val_meta, test_meta


def get_tfidf_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates or loads Hybrid TF-IDF features (Word + Char) for Expert B.
    Fits vectorizers strictly on the Training set.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_tfidf, val_tfidf, test_tfidf) as CSR sparse matrices.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_path = os.path.join(cache_dir, "tfidf_train.npz")
    val_path = os.path.join(cache_dir, "tfidf_val.npz")
    test_path = os.path.join(cache_dir, "tfidf_test.npz")

    files_exist = all(os.path.exists(p) for p in [train_path, val_path, test_path])

    # 1. Try to load from cache
    if load_cached_data and files_exist:
        print("Loading TF-IDF features from cache...")
        try:
            train_tfidf = scipy.sparse.load_npz(train_path)
            val_tfidf = scipy.sparse.load_npz(val_path)
            test_tfidf = scipy.sparse.load_npz(test_path)
            return train_tfidf, val_tfidf, test_tfidf
        except Exception as e:
            print(f"Error loading TF-IDF cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing TF-IDF features...")

    # Prepare text data
    train_text = train_df["text"].fillna("").astype(str)
    val_text = val_df["text"].fillna("").astype(str)
    test_text = test_df["text"].fillna("").astype(str)

    # --- Word N-grams ---
    print(f"Fitting Word TF-IDF (ngram={Config.TFIDF_WORD_NGRAM_RANGE})...")
    # token_pattern=r'\w{1,}' includes single character words (like 'I', 'a')
    word_vectorizer = TfidfVectorizer(
        ngram_range=Config.TFIDF_WORD_NGRAM_RANGE,
        analyzer="word",
        token_pattern=r"\w{1,}",
        dtype=np.float32,
    )
    word_vectorizer.fit(train_text)

    train_word = word_vectorizer.transform(train_text)
    val_word = word_vectorizer.transform(val_text)
    test_word = word_vectorizer.transform(test_text)

    # --- Character N-grams ---
    print(f"Fitting Char TF-IDF (ngram={Config.TFIDF_CHAR_NGRAM_RANGE})...")
    char_vectorizer = TfidfVectorizer(
        ngram_range=Config.TFIDF_CHAR_NGRAM_RANGE, analyzer="char", dtype=np.float32
    )
    char_vectorizer.fit(train_text)

    train_char = char_vectorizer.transform(train_text)
    val_char = char_vectorizer.transform(val_text)
    test_char = char_vectorizer.transform(test_text)

    # --- Stack Features ---
    print("Stacking Word and Char features...")
    # Horizontal stack to combine features for each sample
    train_tfidf = scipy.sparse.hstack([train_word, train_char], format="csr")
    val_tfidf = scipy.sparse.hstack([val_word, val_char], format="csr")
    test_tfidf = scipy.sparse.hstack([test_word, test_char], format="csr")

    # 3. Save to cache
    print(f"Saving TF-IDF features to {cache_dir}...")
    scipy.sparse.save_npz(train_path, train_tfidf)
    scipy.sparse.save_npz(val_path, val_tfidf)
    scipy.sparse.save_npz(test_path, test_tfidf)

    return train_tfidf, val_tfidf, test_tfidf
