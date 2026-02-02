import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
from library.config import Config
from library.utils import compute_qwk
from library.dataset import clean_text


def save_sparse_matrix(filepath, matrix):
    """
    Saves a sparse matrix using numpy .npz format.
    This complies with the requirement to use numpy/parquet and avoid pickle.
    """
    np.savez(
        filepath,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=matrix.shape,
    )


def load_sparse_matrix(filepath):
    """
    Loads a sparse matrix from a numpy .npz file.
    """
    loader = np.load(filepath)
    return csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"]
    )


def train_lexical_fold(train_df, val_df, test_df, fold_idx=0, load_cached_data=True):
    """
    Trains the lexical branch (TF-IDF + Ridge) for a single fold.

    Args:
        train_df (pd.DataFrame): Training data for this fold.
        val_df (pd.DataFrame): Validation data for this fold.
        test_df (pd.DataFrame): Test data.
        fold_idx (int): Fold index used for cache file naming.
        load_cached_data (bool): Whether to attempt loading features from cache.

    Returns:
        model: The trained Ridge regression model.
        val_preds (np.ndarray): Predictions for the validation set.
        test_preds (np.ndarray): Predictions for the test set.
    """
    # Define cache directory
    cache_dir = os.path.join(Config.WORKING_DIR, "lexical_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Append debug suffix if running in debug mode to avoid cache collision
    suffix = "_debug" if Config.DEBUG else ""

    # Define file paths for cached features
    f_X_train = os.path.join(cache_dir, f"fold_{fold_idx}_X_train{suffix}.npz")
    f_X_val = os.path.join(cache_dir, f"fold_{fold_idx}_X_val{suffix}.npz")
    f_X_test = os.path.join(cache_dir, f"fold_{fold_idx}_X_test{suffix}.npz")
    f_y_train = os.path.join(cache_dir, f"fold_{fold_idx}_y_train{suffix}.npy")
    f_y_val = os.path.join(cache_dir, f"fold_{fold_idx}_y_val{suffix}.npy")

    # Check if all cache files exist
    cache_exists = all(
        os.path.exists(p) for p in [f_X_train, f_X_val, f_X_test, f_y_train, f_y_val]
    )

    if load_cached_data and cache_exists:
        # Load from cache
        X_train = load_sparse_matrix(f_X_train)
        X_val = load_sparse_matrix(f_X_val)
        X_test = load_sparse_matrix(f_X_test)
        y_train = np.load(f_y_train)
        y_val = np.load(f_y_val)
    else:
        # Compute features from scratch

        # 1. Clean Text
        # We apply clean_text to ensure consistency with the semantic branch
        train_text = train_df["full_text"].apply(clean_text).tolist()
        val_text = val_df["full_text"].apply(clean_text).tolist()
        test_text = test_df["full_text"].apply(clean_text).tolist()

        # 2. Extract Targets
        y_train = train_df["score"].values.astype(float)
        y_val = val_df["score"].values.astype(float)

        # 3. Vectorization (TF-IDF)
        # Using settings from Config and standard NLP practices
        vectorizer = TfidfVectorizer(
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            min_df=Config.TFIDF_MIN_DF,
            sublinear_tf=True,
            use_idf=True,
            token_pattern=r"(?u)\b\w+\b",  # Capture words including single letters (e.g., 'a', 'I')
        )

        # Fit on training data, transform all
        X_train = vectorizer.fit_transform(train_text)
        X_val = vectorizer.transform(val_text)
        X_test = vectorizer.transform(test_text)

        # 4. Save to Cache
        save_sparse_matrix(f_X_train, X_train)
        save_sparse_matrix(f_X_val, X_val)
        save_sparse_matrix(f_X_test, X_test)
        np.save(f_y_train, y_train)
        np.save(f_y_val, y_val)

    # Train Ridge Regression
    # solver='auto' automatically selects efficient solvers for sparse data (e.g., sparse_cg)
    model = Ridge(alpha=Config.RIDGE_ALPHA, solver="auto", random_state=Config.SEED)
    model.fit(X_train, y_train)

    # Generate Predictions
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    # Evaluation
    qwk = compute_qwk(y_val, val_preds)
    print(f"Fold {fold_idx} Lexical QWK: {qwk}")

    return model, val_preds, test_preds
