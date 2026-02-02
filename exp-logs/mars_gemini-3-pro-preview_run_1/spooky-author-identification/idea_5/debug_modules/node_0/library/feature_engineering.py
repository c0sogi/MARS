import os
import string
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from library.utils import ensure_directory

# Define the working directory for caching
CACHE_DIR = "./working/idea_5/"


def get_tfidf_vectorizer():
    """
    Constructs and returns a Scikit-Learn FeatureUnion combining Word and Character
    TF-IDF vectorizers as specified in the 'Stylometric Residuals' strategy.

    Returns:
        sklearn.pipeline.FeatureUnion: Configured vectorizer pipeline.
    """
    # Word N-grams (1-3)
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"\w{1,}",
    )

    # Character N-grams (3-5)
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    # Combine them
    vectorizer = FeatureUnion(
        [("word_tfidf", word_vectorizer), ("char_tfidf", char_vectorizer)]
    )

    return vectorizer


def extract_meta_features(df, dataset_id, text_col="text", load_cached_data=True):
    """
    Computes explicit meta-features: Sentence Character Length, Word Count, and Punctuation Density.
    Implements caching via Parquet files.

    Args:
        df (pd.DataFrame): Input dataframe containing the text column.
        dataset_id (str): Unique identifier for the dataset (e.g., 'train', 'val', 'test') for cache naming.
        text_col (str): Name of the column containing text.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Dataframe containing only the computed meta-features.
    """
    ensure_directory(CACHE_DIR)
    cache_path = os.path.join(CACHE_DIR, f"meta_features_{dataset_id}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached meta-features for {dataset_id} from {cache_path}...")
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute features
    print(f"Computing meta-features for {dataset_id}...")

    # Ensure text is string and handle NaNs
    texts = df[text_col].fillna("").astype(str)

    meta_df = pd.DataFrame(index=df.index)

    # Character Length
    meta_df["char_len"] = texts.apply(len)

    # Word Count
    meta_df["word_count"] = texts.apply(lambda x: len(x.split()))

    # Punctuation Density
    # count puncts / max(char_len, 1) to avoid div by zero
    punctuation_chars = set(string.punctuation)

    def get_punct_density(text):
        if not text:
            return 0.0
        punct_count = sum(1 for char in text if char in punctuation_chars)
        return punct_count / len(text)

    meta_df["punct_density"] = texts.apply(get_punct_density)

    # 3. Save to cache
    try:
        meta_df.to_parquet(cache_path)
        print(f"Saved meta-features for {dataset_id} to {cache_path}.")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return meta_df


def get_tfidf_features(train_text, val_text, test_text, load_cached_data=True):
    """
    Generates sparse TF-IDF matrices for train, validation, and test sets.
    Fits the vectorizer on the combined corpus to ensure consistent vocabulary.
    Implements caching via NPZ files.

    Args:
        train_text (pd.Series): Training text data.
        val_text (pd.Series): Validation text data.
        test_text (pd.Series): Test text data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_tfidf, X_val_tfidf, X_test_tfidf) as scipy.sparse.csr_matrix.
    """
    ensure_directory(CACHE_DIR)
    path_train = os.path.join(CACHE_DIR, "tfidf_train.npz")
    path_val = os.path.join(CACHE_DIR, "tfidf_val.npz")
    path_test = os.path.join(CACHE_DIR, "tfidf_test.npz")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(path_train)
            and os.path.exists(path_val)
            and os.path.exists(path_test)
        ):
            try:
                print("Loading cached TF-IDF sparse matrices...")
                X_train = scipy.sparse.load_npz(path_train)
                X_val = scipy.sparse.load_npz(path_val)
                X_test = scipy.sparse.load_npz(path_test)
                return X_train, X_val, X_test
            except Exception as e:
                print(f"Failed to load TF-IDF cache: {e}. Recomputing...")

    # 2. Compute features
    print("Fitting TF-IDF Vectorizer on combined corpus...")

    # Handle NaNs
    train_text = train_text.fillna("").astype(str)
    val_text = val_text.fillna("").astype(str)
    test_text = test_text.fillna("").astype(str)

    # Combine for fitting to handle vocabulary fully
    full_corpus = pd.concat([train_text, val_text, test_text], axis=0)

    vectorizer = get_tfidf_vectorizer()
    vectorizer.fit(full_corpus)

    print("Transforming text to TF-IDF features...")
    X_train = vectorizer.transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)

    # 3. Save to cache
    try:
        scipy.sparse.save_npz(path_train, X_train)
        scipy.sparse.save_npz(path_val, X_val)
        scipy.sparse.save_npz(path_test, X_test)
        print("Saved TF-IDF sparse matrices to cache.")
    except Exception as e:
        print(f"Warning: Could not save TF-IDF cache: {e}")

    return X_train, X_val, X_test
