import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
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


def get_structural_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Generates dense structural features via TF-IDF + SVD.
    Cite solution_lesson_node_00012: Multi-Granularity Feature Fusion
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_path = os.path.join(cache_dir, f"train_struct_{Config.SVD_COMPONENTS}.npy")
    val_path = os.path.join(cache_dir, f"val_struct_{Config.SVD_COMPONENTS}.npy")
    test_path = os.path.join(cache_dir, f"test_struct_{Config.SVD_COMPONENTS}.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    )

    if load_cached_data and cache_exists:
        print("Loading structural features from cache...")
        X_train = np.load(train_path)
        X_val = np.load(val_path)
        X_test = np.load(test_path)

        if X_train.shape[0] == len(train_df):
            return X_train, X_val, X_test
        else:
            print("Dimension mismatch. Regenerating...")

    print("Generating structural features (TF-IDF + SVD)...")

    # Use clean_comment for structural features (NBSVM logic)
    col = "clean_comment" if "clean_comment" in train_df.columns else "Comment"

    train_text = train_df[col].fillna("").astype(str)
    val_text = val_df[col].fillna("").astype(str)
    test_text = test_df[col].fillna("").astype(str)

    # 1. Word N-grams (1-2)
    vec_word = TfidfVectorizer(
        ngram_range=Config.NBSVM_WORD_NGRAM_RANGE,
        min_df=3,
        max_df=0.9,
        strip_accents="unicode",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )
    X_train_word = vec_word.fit_transform(train_text)
    X_val_word = vec_word.transform(val_text)
    X_test_word = vec_word.transform(test_text)

    # 2. Char N-grams (3-5)
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
    X_train_char = vec_char.fit_transform(train_text)
    X_val_char = vec_char.transform(val_text)
    X_test_char = vec_char.transform(test_text)

    # Stack
    X_train_sparse = hstack([X_train_word, X_train_char])
    X_val_sparse = hstack([X_val_word, X_val_char])
    X_test_sparse = hstack([X_test_word, X_test_char])

    # SVD Projection
    print(f"  Projecting to {Config.SVD_COMPONENTS} dimensions via SVD...")
    svd = TruncatedSVD(n_components=Config.SVD_COMPONENTS, random_state=Config.SEED)
    X_train_dense = svd.fit_transform(X_train_sparse)
    X_val_dense = svd.transform(X_val_sparse)
    X_test_dense = svd.transform(X_test_sparse)

    # Save
    np.save(train_path, X_train_dense)
    np.save(val_path, X_val_dense)
    np.save(test_path, X_test_dense)

    return X_train_dense, X_val_dense, X_test_dense


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
