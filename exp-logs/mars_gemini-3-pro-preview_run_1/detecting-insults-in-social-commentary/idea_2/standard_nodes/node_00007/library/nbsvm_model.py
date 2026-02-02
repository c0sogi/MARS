import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from library.config import Config


class NBSVM:
    """
    NBSVM (Naive Bayes - Support Vector Machine) implementation.
    Actually uses Logistic Regression with Naive Bayes features (NB-LR).
    """

    def __init__(self, C=1.0, max_iter=1000, seed=42):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="liblinear",  # Good for sparse data
            random_state=self.seed,
        )
        self.r = None

    def fit(self, X, y):
        """
        Fit the NBSVM model.
        X: Sparse matrix of features
        y: Target labels
        """
        # Ensure X is CSR for efficient slicing
        if not isinstance(X, csr_matrix):
            X = X.tocsr()

        y = np.array(y)

        # Calculate Log-Count Ratios (r)
        # p: sum of feature counts for class 1
        # q: sum of feature counts for class 0
        # Add 1 for smoothing (Laplace)

        # Summing sparse matrix over axis 0 returns a dense matrix (1, n_features)
        p = X[y == 1].sum(axis=0) + 1
        q = X[y == 0].sum(axis=0) + 1

        # Normalize (L1 norm)
        p_norm = p / np.sum(p)
        q_norm = q / np.sum(q)

        # Log ratio
        self.r = np.log(p_norm / q_norm)

        # Scale features
        # multiply acts element-wise. r is (1, n_features), broadcasts over rows
        X_nb = X.multiply(self.r)

        # Train Logistic Regression
        self.model.fit(X_nb, y)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        if not isinstance(X, csr_matrix):
            X = X.tocsr()

        # Scale features using learned r
        X_nb = X.multiply(self.r)

        # Return probability of positive class
        return self.model.predict_proba(X_nb)[:, 1]


def get_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates or loads TF-IDF features for NBSVM.
    Implements caching using .npz files.
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_path = os.path.join(cache_dir, "nbsvm_features_train.npz")
    val_path = os.path.join(cache_dir, "nbsvm_features_val.npz")
    test_path = os.path.join(cache_dir, "nbsvm_features_test.npz")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    )

    if load_cached_data and cache_exists:
        print("Loading NBSVM features from cache...")
        X_train = load_npz(train_path)
        X_val = load_npz(val_path)
        X_test = load_npz(test_path)

        # Cite debug_lesson_1: Validate Cache Integrity to Avoid Stale Data Mismatches
        if (
            X_train.shape[0] != len(train_df)
            or X_val.shape[0] != len(val_df)
            or X_test.shape[0] != len(test_df)
        ):
            print(
                "Dimension mismatch detected between cached features and current data. Regenerating features..."
            )
        else:
            return X_train, X_val, X_test

    print("Generating NBSVM features (TF-IDF)...")

    # Prepare text
    # Use 'clean_comment' if available (processed by data_loader), else 'Comment'
    col = "clean_comment" if "clean_comment" in train_df.columns else "Comment"

    train_text = train_df[col].fillna("").astype(str)
    val_text = val_df[col].fillna("").astype(str)
    test_text = test_df[col].fillna("").astype(str)

    # 1. Word N-grams
    print(f"  Fitting Word Vectorizer {Config.NBSVM_WORD_NGRAM_RANGE}...")
    vec_word = TfidfVectorizer(
        ngram_range=Config.NBSVM_WORD_NGRAM_RANGE,
        min_df=3,
        max_df=0.9,
        strip_accents="unicode",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )
    # Fit on train, transform all
    X_train_word = vec_word.fit_transform(train_text)
    X_val_word = vec_word.transform(val_text)
    X_test_word = vec_word.transform(test_text)

    # 2. Char N-grams
    print(f"  Fitting Char Vectorizer {Config.NBSVM_CHAR_NGRAM_RANGE}...")
    vec_char = TfidfVectorizer(
        ngram_range=Config.NBSVM_CHAR_NGRAM_RANGE,
        min_df=3,
        max_df=0.9,
        strip_accents="unicode",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
        analyzer="char",
    )
    # Fit on train, transform all
    X_train_char = vec_char.fit_transform(train_text)
    X_val_char = vec_char.transform(val_text)
    X_test_char = vec_char.transform(test_text)

    # Concatenate
    print("  Stacking features...")
    X_train = hstack([X_train_word, X_train_char])
    X_val = hstack([X_val_word, X_val_char])
    X_test = hstack([X_test_word, X_test_char])

    # Save to cache
    print("Saving NBSVM features to cache...")
    save_npz(train_path, X_train)
    save_npz(val_path, X_val)
    save_npz(test_path, X_test)

    return X_train, X_val, X_test


def run_nbsvm(train_df, val_df, test_df, load_cached_data=True):
    """
    Main function to run the NBSVM branch.
    1. Gets features.
    2. Trains model.
    3. Evaluates on validation.
    4. Generates test predictions.
    """
    # 1. Feature Extraction
    X_train, X_val, X_test = get_features(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    y_train = train_df["Insult"].values
    y_val = val_df["Insult"].values

    # 2. Model Training
    print("Training NBSVM Model...")
    model = NBSVM(C=Config.NBSVM_C, max_iter=Config.NBSVM_MAX_ITER, seed=Config.SEED)
    model.fit(X_train, y_train)

    # 3. Validation
    print("Evaluating NBSVM on Validation Set...")
    val_preds = model.predict_proba(X_val)
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"NBSVM Validation AUC: {val_auc}")

    # 4. Test Prediction
    print("Generating Test Predictions...")
    test_preds = model.predict_proba(X_test)

    return val_preds, test_preds
