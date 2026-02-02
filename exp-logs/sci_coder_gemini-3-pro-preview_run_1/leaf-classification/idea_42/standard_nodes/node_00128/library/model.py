import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from scipy.special import softmax
from library.config import Config
from library.utils import ensure_float64


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    A Custom Linear Discriminant Classifier using the Oracle Approximating Shrinkage (OAS)
    estimator for the shared covariance matrix.

    This implementation explicitly uses float64 precision for all linear algebra operations
    to prevent spectral truncation errors and avoid the 1e-7 metric floor in log-loss.

    The decision function is linearized as:
        Z_k(x) = x^T (P mu_k) - 0.5 * (mu_k^T P mu_k) + log(pi_k)
    Where:
        P: Precision matrix (inverse covariance)
        mu_k: Mean vector for class k
        pi_k: Prior probability for class k
    """

    def __init__(self, assume_centered=True):
        """
        Args:
            assume_centered (bool): If True, data will be centered manually before passing
                                    to the OAS estimator. Defaults to True as per strategy.
        """
        self.assume_centered = assume_centered

        # Model parameters
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.covariance_ = None
        self.precision_ = None

        # Linearized decision boundary parameters
        self.W_ = None  # Shape (n_classes, n_features)
        self.b_ = None  # Shape (n_classes,)

    def fit(self, X, y):
        """
        Fits the OAS Discriminant model.

        1. Computes empirical class means and priors.
        2. Centers the data (residuals).
        3. Estimates covariance using OAS.
        4. Derives weights (W) and bias (b) for linear inference.

        Args:
            X (array-like): Training features.
            y (array-like): Training labels (strings or ints).

        Returns:
            self
        """
        # 1. Enforce Precision
        X = ensure_float64(X)
        y = np.array(y)  # Keep as is for class extraction

        # 2. Encode Classes
        # Sort classes alphabetically to ensure deterministic ordering matching submission format
        self.classes_ = np.unique(y)
        self.classes_.sort()
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Map labels to indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[yi] for yi in y])

        # 3. Compute Priors and Means
        self.means_ = np.zeros((n_classes, n_features), dtype=Config.FLOAT_TYPE)
        self.priors_ = np.zeros(n_classes, dtype=Config.FLOAT_TYPE)

        # We also need centered data for OAS
        X_centered = np.zeros_like(X, dtype=Config.FLOAT_TYPE)

        for k in range(n_classes):
            mask = y_idx == k
            X_k = X[mask]

            # Empirical Prior
            self.priors_[k] = X_k.shape[0] / n_samples

            # Empirical Mean
            mean_k = np.mean(X_k, axis=0)
            self.means_[k] = mean_k

            # Center the data for this class
            X_centered[mask] = X_k - mean_k

        # 4. Estimate Covariance via OAS
        # We use the residuals (X - mu_y) to estimate the shared covariance
        oas = OAS(assume_centered=self.assume_centered)
        oas.fit(X_centered)

        self.covariance_ = ensure_float64(oas.covariance_)
        self.precision_ = ensure_float64(oas.precision_)

        # 5. Derive Linear Weights and Bias
        # W_k = P * mu_k
        # b_k = -0.5 * (mu_k^T * W_k) + log(pi_k)

        # W shape: (n_classes, n_features)
        # means_.T shape: (n_features, n_classes)
        # precision_ shape: (n_features, n_features)
        # W^T = P * means_.T -> W = (P * means_.T).T = means_ * P
        # Note: precision_ is symmetric, so P = P^T.

        self.W_ = np.dot(self.means_, self.precision_)

        # Compute bias terms
        # Element-wise multiplication and sum over features for the quadratic term
        # (mu_k dot W_k)
        quadratic_term = np.sum(self.means_ * self.W_, axis=1)
        self.b_ = -0.5 * quadratic_term + np.log(self.priors_)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linearized formulation.

        Z = X W^T + b
        P = Softmax(Z)

        Args:
            X (array-like): Test features.

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        # Enforce Precision
        X = ensure_float64(X)

        # Linear Score (Logits)
        # X: (n_samples, n_features)
        # W_: (n_classes, n_features) -> W_.T: (n_features, n_classes)
        # b_: (n_classes,)
        logits = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax
        # scipy.special.softmax is numerically stable
        probas = softmax(logits, axis=1)

        return ensure_float64(probas)

    def predict(self, X):
        """
        Predicts class labels.

        Args:
            X (array-like): Test features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.classes_[indices]
