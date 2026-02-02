import os
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from library.config import Config
from library.utils import seed_everything, compute_metric
from library.data_manager import load_raw_data, LABEL_MAP


def save_sparse_csr(filename, array):
    """
    Saves a CSR sparse matrix to disk as separate .npy files for
    data, indices, indptr, and shape.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    np.save(filename + "_data.npy", array.data)
    np.save(filename + "_indices.npy", array.indices)
    np.save(filename + "_indptr.npy", array.indptr)
    np.save(filename + "_shape.npy", array.shape)


def load_sparse_csr(filename):
    """
    Loads a CSR sparse matrix from disk.
    """
    data = np.load(filename + "_data.npy")
    indices = np.load(filename + "_indices.npy")
    indptr = np.load(filename + "_indptr.npy")
    shape = np.load(filename + "_shape.npy")

    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def get_tfidf_features(train_text, val_text, test_text, load_cached_data=True):
    """
    Generates or loads TF-IDF features for the statistical branch.
    Combines Word N-grams and Character N-grams.
    """
    cache_dir = os.path.join(Config.WORKING_DIR, "tfidf_features")
    os.makedirs(cache_dir, exist_ok=True)

    # File prefixes
    train_cache = os.path.join(cache_dir, "X_train_tfidf")
    val_cache = os.path.join(cache_dir, "X_val_tfidf")
    test_cache = os.path.join(cache_dir, "X_test_tfidf")

    # Check if all component files exist
    files_exist = True
    for prefix in [train_cache, val_cache, test_cache]:
        for suffix in ["_data.npy", "_indices.npy", "_indptr.npy", "_shape.npy"]:
            if not os.path.exists(prefix + suffix):
                files_exist = False
                break

    if load_cached_data and files_exist:
        print("Loading cached TF-IDF features...")
        try:
            X_train = load_sparse_csr(train_cache)
            X_val = load_sparse_csr(val_cache)
            X_test = load_sparse_csr(test_cache)
            return X_train, X_val, X_test
        except Exception as e:
            print(f"Error loading cache ({e}). Recomputing...")

    print("Computing TF-IDF features from scratch...")

    # 1. Word N-grams
    print("  Fitting Word Vectorizer...")
    word_vectorizer = TfidfVectorizer(
        ngram_range=Config.TFIDF_WORD_NGRAM_RANGE,
        max_features=Config.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        analyzer="word",
        token_pattern=r"\w{1,}",
        strip_accents="unicode",
    )
    word_vectorizer.fit(train_text)

    train_word = word_vectorizer.transform(train_text)
    val_word = word_vectorizer.transform(val_text)
    test_word = word_vectorizer.transform(test_text)

    # 2. Char N-grams
    print("  Fitting Char Vectorizer...")
    char_vectorizer = TfidfVectorizer(
        ngram_range=Config.TFIDF_CHAR_NGRAM_RANGE,
        max_features=Config.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        analyzer="char",
        strip_accents="unicode",
    )
    char_vectorizer.fit(train_text)

    train_char = char_vectorizer.transform(train_text)
    val_char = char_vectorizer.transform(val_text)
    test_char = char_vectorizer.transform(test_text)

    # 3. Combine
    print("  Stacking features...")
    X_train = sparse.hstack([train_word, train_char], format="csr")
    X_val = sparse.hstack([val_word, val_char], format="csr")
    X_test = sparse.hstack([test_word, test_char], format="csr")

    # 4. Cache
    print(f"  Saving features to {cache_dir}...")
    save_sparse_csr(train_cache, X_train)
    save_sparse_csr(val_cache, X_val)
    save_sparse_csr(test_cache, X_test)

    return X_train, X_val, X_test


def train_tfidf_models(X_train, y_train, X_val, X_test):
    """
    Trains Logistic Regression and Naive Bayes models.
    Returns the models and their predictions on Val and Test sets.
    """
    print("Training Statistical Models...")

    # Logistic Regression
    print("  Training Logistic Regression...")
    lr_model = LogisticRegression(
        C=1.0,
        solver="sag",
        multi_class="multinomial",
        max_iter=1000,
        random_state=Config.SEED,
        n_jobs=-1,
    )
    lr_model.fit(X_train, y_train)

    lr_val_preds = lr_model.predict_proba(X_val)
    lr_test_preds = lr_model.predict_proba(X_test)

    # Multinomial Naive Bayes
    print("  Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.01)  # Low alpha usually works well for text
    nb_model.fit(X_train, y_train)

    nb_val_preds = nb_model.predict_proba(X_val)
    nb_test_preds = nb_model.predict_proba(X_test)

    return (
        (lr_model, nb_model),
        (lr_val_preds, nb_val_preds),
        (lr_test_preds, nb_test_preds),
    )


def optimize_stat_blend(lr_preds, nb_preds, y_true):
    """
    Finds the optimal weight alpha for: P = alpha * LR + (1-alpha) * NB
    Minimizing Log Loss on the validation set.
    """
    print("Optimizing Statistical Blend Weight...")

    def loss_func(alpha):
        # Constrain alpha to [0, 1] effectively
        if alpha < 0 or alpha > 1:
            return 100.0

        blended = alpha * lr_preds + (1 - alpha) * nb_preds
        return compute_metric(y_true, blended)

    # Use scalar minimization
    res = minimize_scalar(loss_func, bounds=(0, 1), method="bounded")
    best_alpha = res.x
    best_loss = res.fun

    print(f"  Optimal Alpha (LR weight): {best_alpha:.6f}")
    print(f"  Best Validation Log Loss: {best_loss}")  # Full precision

    return best_alpha


def run_statistical_pipeline(load_cached_data=True, debug=False):
    """
    Main entry point for the statistical branch.

    Returns:
        tuple: (val_preds_blended, test_preds_blended, best_alpha)
    """
    seed_everything()

    # 1. Load Data
    train_df, val_df, test_df = load_raw_data(debug=debug)

    # Prepare text and labels
    train_text = train_df["text"].fillna("").tolist()
    val_text = val_df["text"].fillna("").tolist()
    test_text = test_df["text"].fillna("").tolist()

    # Map labels to integers
    y_train = train_df["author"].map(LABEL_MAP).values
    y_val = val_df["author"].map(LABEL_MAP).values

    # 2. Feature Extraction
    X_train, X_val, X_test = get_tfidf_features(
        train_text, val_text, test_text, load_cached_data=load_cached_data
    )

    # 3. Train Models
    models, val_preds_tuple, test_preds_tuple = train_tfidf_models(
        X_train, y_train, X_val, X_test
    )
    lr_val, nb_val = val_preds_tuple
    lr_test, nb_test = test_preds_tuple

    # 4. Optimize Blend
    best_alpha = optimize_stat_blend(lr_val, nb_val, y_val)

    # 5. Create Final Blended Predictions
    val_preds_blended = best_alpha * lr_val + (1 - best_alpha) * nb_val
    test_preds_blended = best_alpha * lr_test + (1 - best_alpha) * nb_test

    return val_preds_blended, test_preds_blended, best_alpha
