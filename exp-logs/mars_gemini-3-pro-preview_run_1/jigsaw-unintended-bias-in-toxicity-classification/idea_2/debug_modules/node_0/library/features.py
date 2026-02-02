import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import Config


def get_configured_vectorizers():
    """
    Initializes and returns the word and character TF-IDF vectorizers
    based on the configuration settings.

    Returns:
        tuple: (word_vectorizer, char_vectorizer)
    """
    # Word-level TF-IDF Vectorizer
    word_vectorizer = TfidfVectorizer(
        ngram_range=Config.WORD_NGRAM_RANGE,
        max_features=Config.WORD_MAX_FEATURES,
        min_df=Config.WORD_MIN_DF,
        analyzer="word",
        token_pattern=r"\w{1,}",
        strip_accents="unicode",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )

    # Character-level TF-IDF Vectorizer
    char_vectorizer = TfidfVectorizer(
        ngram_range=Config.CHAR_NGRAM_RANGE,
        max_features=Config.CHAR_MAX_FEATURES,
        min_df=Config.CHAR_MIN_DF,
        analyzer="char",
        strip_accents="unicode",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )

    return word_vectorizer, char_vectorizer


def build_feature_matrix(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    load_cached_data: bool = True,
):
    """
    Constructs the feature matrices for train, validation, and test sets.
    Uses caching to avoid re-computation.

    Args:
        train_df: Training DataFrame with text column.
        val_df: Validation DataFrame with text column.
        test_df: Test DataFrame with text column.
        load_cached_data: Whether to attempt loading from disk.

    Returns:
        tuple: (X_train, X_val, X_test) as scipy sparse matrices.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_feat_path = Config.TRAIN_FEATURES_PATH
    val_feat_path = Config.VAL_FEATURES_PATH
    test_feat_path = Config.TEST_FEATURES_PATH

    X_train = None
    X_val = None
    X_test = None

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_feat_path)
            and os.path.exists(val_feat_path)
            and os.path.exists(test_feat_path)
        ):
            try:
                # print("Loading features from cache...")
                X_train = sparse.load_npz(train_feat_path)
                X_val = sparse.load_npz(val_feat_path)
                X_test = sparse.load_npz(test_feat_path)

                # Basic validation to ensure dimensions match current dataframes
                # (Assuming row counts haven't changed if cache exists)
                if (
                    X_train.shape[0] == len(train_df)
                    and X_val.shape[0] == len(val_df)
                    and X_test.shape[0] == len(test_df)
                ):
                    return X_train, X_val, X_test
                else:
                    # print("Cache dimension mismatch. Recomputing...")
                    X_train, X_val, X_test = None, None, None
            except Exception:
                # print("Error loading cache. Recomputing...")
                X_train, X_val, X_test = None, None, None

    # 2. Compute if needed
    if X_train is None:
        # print("Computing TF-IDF features...")

        # Extract text
        train_text = train_df[Config.TEXT_COL]
        val_text = val_df[Config.TEXT_COL]
        test_text = test_df[Config.TEXT_COL]

        # Get vectorizers
        word_vec, char_vec = get_configured_vectorizers()

        # Fit on Train and Transform
        # print("Fitting Word Vectorizer...")
        word_vec.fit(train_text)
        train_word = word_vec.transform(train_text)
        val_word = word_vec.transform(val_text)
        test_word = word_vec.transform(test_text)

        # print("Fitting Char Vectorizer...")
        char_vec.fit(train_text)
        train_char = char_vec.transform(train_text)
        val_char = char_vec.transform(val_text)
        test_char = char_vec.transform(test_text)

        # Stack features
        # print("Stacking features...")
        X_train = sparse.hstack([train_word, train_char]).tocsr()
        X_val = sparse.hstack([val_word, val_char]).tocsr()
        X_test = sparse.hstack([test_word, test_char]).tocsr()

        # Save to cache
        # print("Saving features to cache...")
        sparse.save_npz(train_feat_path, X_train)
        sparse.save_npz(val_feat_path, X_val)
        sparse.save_npz(test_feat_path, X_test)

    return X_train, X_val, X_test
