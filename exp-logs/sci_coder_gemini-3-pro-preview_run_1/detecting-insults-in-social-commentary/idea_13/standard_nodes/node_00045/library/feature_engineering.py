import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from library.config import Config
from library.utils import get_logger

logger = get_logger("feature_engineering")


def generate_svd_features(
    train_text, val_text, test_text, fold_idx, load_cached_data=True
):
    """
    Generates SVD features from text data using TF-IDF and TruncatedSVD.
    Strictly fits transformers ONLY on the training split to prevent data leakage.

    Args:
        train_text (list or pd.Series): Text data for the training split.
        val_text (list or pd.Series): Text data for the validation split.
        test_text (list or pd.Series): Text data for the test set.
        fold_idx (int): The current fold index (used for cache naming).
        load_cached_data (bool): If True, attempts to load features from cache.

    Returns:
        tuple: (train_svd, val_svd, test_svd) as numpy arrays.
    """
    # Ensure cache directory exists
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    train_cache_path = os.path.join(cache_dir, f"fold_{fold_idx}_train_svd.npy")
    val_cache_path = os.path.join(cache_dir, f"fold_{fold_idx}_val_svd.npy")
    test_cache_path = os.path.join(cache_dir, f"fold_{fold_idx}_test_svd.npy")

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            logger.info(f"Loading SVD features from cache for fold {fold_idx}...")
            try:
                train_svd = np.load(train_cache_path)
                val_svd = np.load(val_cache_path)
                test_svd = np.load(test_cache_path)
                return train_svd, val_svd, test_svd
            except Exception as e:
                logger.warning(
                    f"Failed to load cache for fold {fold_idx}: {e}. Recomputing..."
                )
        else:
            logger.info(f"Cache miss for fold {fold_idx}. Generating features...")
    else:
        logger.info(f"Ignoring cache. Generating features for fold {fold_idx}...")

    # Preprocessing: Handle NaNs and ensure string format
    logger.info("Preprocessing text data...")
    train_text = pd.Series(train_text).fillna("").astype(str)
    val_text = pd.Series(val_text).fillna("").astype(str)
    test_text = pd.Series(test_text).fillna("").astype(str)

    # 1. Word N-grams TF-IDF
    logger.info(f"Fitting Word TF-IDF (range={Config.tfidf_word_ngram_range})...")
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=Config.tfidf_word_ngram_range,
        min_df=2,
        sublinear_tf=True,
        use_idf=True,
    )
    # Fit strictly on training data
    train_word_tfidf = word_vectorizer.fit_transform(train_text)
    val_word_tfidf = word_vectorizer.transform(val_text)
    test_word_tfidf = word_vectorizer.transform(test_text)

    # 2. Character N-grams TF-IDF
    logger.info(f"Fitting Char TF-IDF (range={Config.tfidf_char_ngram_range})...")
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=Config.tfidf_char_ngram_range,
        min_df=2,
        sublinear_tf=True,
        use_idf=True,
    )
    # Fit strictly on training data
    train_char_tfidf = char_vectorizer.fit_transform(train_text)
    val_char_tfidf = char_vectorizer.transform(val_text)
    test_char_tfidf = char_vectorizer.transform(test_text)

    # 3. Combine Features
    logger.info("Concatenating Word and Char TF-IDF matrices...")
    train_tfidf = sparse.hstack([train_word_tfidf, train_char_tfidf])
    val_tfidf = sparse.hstack([val_word_tfidf, val_char_tfidf])
    test_tfidf = sparse.hstack([test_word_tfidf, test_char_tfidf])

    # 4. Truncated SVD
    logger.info(f"Fitting TruncatedSVD (n_components={Config.svd_components})...")
    svd = TruncatedSVD(n_components=Config.svd_components, random_state=Config.seed)

    # Fit strictly on training data
    train_svd = svd.fit_transform(train_tfidf)
    val_svd = svd.transform(val_tfidf)
    test_svd = svd.transform(test_tfidf)

    # 5. Save to Cache
    logger.info(f"Saving generated features to {cache_dir}...")
    np.save(train_cache_path, train_svd)
    np.save(val_cache_path, val_svd)
    np.save(test_cache_path, test_svd)

    return train_svd, val_svd, test_svd
