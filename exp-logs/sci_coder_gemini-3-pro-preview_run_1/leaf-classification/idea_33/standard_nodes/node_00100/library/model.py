import numpy as np
import scipy.linalg
import scipy.special
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from library.data_loader import load_data
from library.utils import save_submission


class ExactOASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using OAS Covariance and Exact Precision Matrix.
    Implements the 'Alphanumeric Exact-Precision OAS Discriminant'.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.precision_ = None
        self.weights_ = None
        self.bias_ = None
        self.priors_ = None
        self.le_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation and explicit precision matrix.
        Operations performed in float64.

        Args:
            X (array-like): Training data (n_samples, n_features).
            y (array-like): Target labels.
        """
        # Ensure float64 precision
        X = np.array(X, dtype=np.float64)

        # Encode labels
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Parameter Estimation
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        # Compute class means and priors
        for k in range(n_classes):
            X_k = X[y_enc == k]
            self.means_[k] = np.mean(X_k, axis=0)
            self.priors_[k] = len(X_k) / len(X)

        # Compute residuals (centered data)
        # We assume a shared covariance matrix across classes (LDA assumption)
        residuals = X - self.means_[y_enc]

        # Estimate Covariance using OAS
        # assume_centered=True because we manually centered using class means (Cite 00061)
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        # Use precision_ directly from OAS (Cite 00062)
        # This uses SVD-based pseudo-inverse internally which is more robust than Cholesky for ill-conditioned matrices
        self.precision_ = oas.precision_.astype(np.float64)

        # 2. Exact Weight Derivation
        # W = Sigma^-1 * Means.T  => W = Precision * Means.T
        # But we store W such that logits = X @ W.T + b
        # So W = (Precision @ Means.T).T = Means @ Precision.T = Means @ Precision (symmetric)
        self.weights_ = self.means_ @ self.precision_

        # 3. Compute Bias
        # b_k = -0.5 * (mu_k . W_k) + log(pi_k)
        term1 = -0.5 * np.sum(self.means_ * self.weights_, axis=1)
        term2 = np.log(self.priors_)
        self.bias_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predicts probabilities using linearized inference with extended precision.

        Args:
            X (array-like): Test data.

        Returns:
            np.ndarray: Probabilities (n_samples, n_classes).
        """
        # Use longdouble for inference precision (Cite 00057 extension)
        X = np.array(X, dtype=np.longdouble)
        W = self.weights_.astype(np.longdouble)
        b = self.bias_.astype(np.longdouble)

        # Linear Scoring: Z = X * W.T + b
        logits = X @ W.T + b

        # Manual Softmax in longdouble for maximum precision
        # shift logits for stability
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Cast back to float64 for compatibility and clipping
        probs = probs.astype(np.float64)

        # Clip to avoid log extremes as per metric spec
        # range [1e-15, 1 - 1e-15]
        probs = np.clip(probs, 1e-15, 1 - 1e-15)

        return probs


def run_pipeline(
    cache_dir="./working/idea_33/", submission_path="./submission/submission.csv"
):
    """
    Orchestrates the data loading, training, validation, and submission generation.
    """
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_data(
        cache_dir=cache_dir
    )

    print("Initializing Alphanumeric Exact-Precision OAS Discriminant...")
    model = ExactOASDiscriminant()

    print("Fitting model...")
    model.fit(X_train, y_train)

    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    loss = log_loss(y_val, val_probs, labels=model.classes_)
    print(f"Validation Multi-class Log Loss: {loss:.15f}")

    print("Generating predictions for Test Set...")
    test_probs = model.predict_proba(X_test)

    print(f"Saving submission to {submission_path}...")
    save_submission(test_ids, model.classes_, test_probs, submission_path)

    return model
