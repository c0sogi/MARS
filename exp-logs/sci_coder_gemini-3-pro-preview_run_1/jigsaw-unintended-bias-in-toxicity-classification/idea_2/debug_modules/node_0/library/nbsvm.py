import os
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data_utils import load_data
from library.features import build_feature_matrix
from library.weighting import compute_sample_weights


class NBSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    NBSVM (Naive Bayes - Support Vector Machine) Classifier.

    This model combines Naive Bayes feature scaling with a Logistic Regression classifier
    (which acts as the 'SVM' linear separator in this context). It effectively handles
    text classification baselines.
    """

    def __init__(
        self, C=1.0, solver="lbfgs", max_iter=1000, n_jobs=-1, random_state=42
    ):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.clf = None
        self._r = None

    def _compute_nb_ratios(self, X, y):
        """
        Computes the Log-Count Ratios (r) for Naive Bayes scaling.
        r = log(p/q)
        where p is prob of feature in positive class, q in negative class.
        """
        # Check if X is sparse
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        # Identify indices for positive and negative classes
        p_indices = np.where(y == 1)[0]
        n_indices = np.where(y == 0)[0]

        # Calculate sum of features for each class
        # Note: Summing sparse matrix over axis 0 returns a dense matrix/array
        p_sum = X[p_indices].sum(axis=0)
        n_sum = X[n_indices].sum(axis=0)

        # Convert to 1D array
        p_sum = np.asarray(p_sum).flatten()
        n_sum = np.asarray(n_sum).flatten()

        # Apply Laplace smoothing (alpha=1)
        p_sum = p_sum + 1.0
        n_sum = n_sum + 1.0

        # Normalize to get probabilities
        p_prob = p_sum / np.sum(p_sum)
        n_prob = n_sum / np.sum(n_sum)

        # Calculate log-ratio
        r = np.log(p_prob / n_prob)

        return r

    def fit(self, X, y, sample_weight=None):
        """
        Fits the NBSVM model.

        Args:
            X: Sparse feature matrix.
            y: Binary target vector.
            sample_weight: Optional weights for samples (used for bias mitigation).
        """
        # Compute Naive Bayes ratios
        self._r = self._compute_nb_ratios(X, y)

        # Scale features: Element-wise multiplication of X by r
        # sparse.multiply broadcasts the array over rows
        X_scaled = X.multiply(self._r)

        # Initialize Logistic Regression
        self.clf = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0,  # Keep silent as per requirements
        )

        # Fit the classifier
        self.clf.fit(X_scaled, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities.
        """
        if self.clf is None or self._r is None:
            raise RuntimeError("Model must be fitted before prediction.")

        # Scale features using the learned ratios
        X_scaled = X.multiply(self._r)

        # Return probabilities
        return self.clf.predict_proba(X_scaled)

    def predict(self, X):
        """
        Predicts class labels.
        """
        if self.clf is None or self._r is None:
            raise RuntimeError("Model must be fitted before prediction.")

        X_scaled = X.multiply(self._r)
        return self.clf.predict(X_scaled)


def train_nbsvm(load_cached_data=True):
    """
    Training pipeline for the NBSVM model with Bias-Centric Sample Weighting.
    """
    # 1. Load Data
    train_df = load_data("train", load_cached_data=load_cached_data)
    val_df = load_data("val", load_cached_data=load_cached_data)
    test_df = load_data("test", load_cached_data=load_cached_data)

    # 2. Build Features
    X_train, X_val, X_test = build_feature_matrix(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 3. Prepare Targets
    y_train = train_df[Config.BINARY_TARGET_COL].values
    y_val = val_df[Config.BINARY_TARGET_COL].values

    # 4. Compute Sample Weights (Bias Mitigation)
    sample_weights = compute_sample_weights(train_df, load_cached_data=load_cached_data)

    # 5. Initialize Model
    model = NBSVMClassifier(
        C=Config.C,
        solver=Config.SOLVER,
        max_iter=Config.MAX_ITER,
        n_jobs=Config.N_JOBS,
        random_state=Config.SEED,
    )

    # 6. Train
    print("Training NBSVM model...")
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # 7. Evaluate
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_probs)

    print(f"Validation Overall ROC-AUC: {auc_score}")

    # 8. Save Model
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    joblib.dump(model, Config.MODEL_PATH)
    print(f"Model saved to {Config.MODEL_PATH}")

    return model


def generate_submission(load_cached_data=True):
    """
    Generates the submission file using the trained NBSVM model.
    """
    # 1. Load Data & Features
    # We need the test dataframe for IDs and the test features for prediction
    test_df = load_data("test", load_cached_data=load_cached_data)

    # We need to ensure features are loaded/built.
    # Calling build_feature_matrix retrieves the cached X_test if available.
    # We pass empty train/val dfs if we only want to retrieve cache,
    # but strictly we should pass the real ones or rely on the cache existing from training step.
    # Assuming training has run, cache exists.
    # To be safe, we reload all, relying on cache speed.
    train_df = load_data("train", load_cached_data=load_cached_data)
    val_df = load_data("val", load_cached_data=load_cached_data)
    _, _, X_test = build_feature_matrix(
        train_df, val_df, test_df, load_cached_data=load_cached_data
    )

    # 2. Load Model
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Run training first."
        )

    model = joblib.load(Config.MODEL_PATH)

    # 3. Predict
    print("Generating predictions for Test Set...")
    test_probs = model.predict_proba(X_test)[:, 1]

    # 4. Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_df["id"], "prediction": test_probs})

    # 5. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
