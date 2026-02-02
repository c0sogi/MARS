import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from library.config import NUMERIC_DTYPE, OAS_ASSUME_CENTERED


class HighPrecisionOAS(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS)
    for covariance estimation. Designed for high-precision (float64) inference.

    This model assumes a common covariance matrix across all classes (Linear Discriminant Analysis)
    but uses OAS to robustly estimate this covariance in high-dimensional settings where
    standard empirical covariance is ill-conditioned.

    The decision function is linearized:
        Z_k = x^T W_k + b_k
    where:
        W_k = \Sigma^{-1} \mu_k
        b_k = -0.5 \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k)
    """

    def __init__(self, assume_centered=OAS_ASSUME_CENTERED):
        """
        Args:
            assume_centered (bool): If True, data will not be centered before computation.
                                    We manually center residuals for covariance estimation,
                                    so this should typically be True for the internal OAS estimator.
        """
        self.assume_centered = assume_centered

        # Model parameters
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.covariance_ = None
        self.precision_ = None

        # Linear decision boundary parameters
        self.coef_ = None  # Shape: (n_classes, n_features)
        self.intercept_ = None  # Shape: (n_classes,)

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features).
                            Must be float64.
            y (np.ndarray): Target labels, shape (n_samples,).

        Returns:
            self
        """
        # Enforce double precision
        X = X.astype(NUMERIC_DTYPE, copy=False)

        # Identify classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Initialize statistics
        self.means_ = np.zeros((n_classes, n_features), dtype=NUMERIC_DTYPE)
        self.priors_ = np.zeros(n_classes, dtype=NUMERIC_DTYPE)

        # 1. Compute Empirical Class Means and Priors
        # We calculate the arithmetic mean for each class.
        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.means_[idx, :] = np.mean(X_cls, axis=0, dtype=NUMERIC_DTYPE)
            self.priors_[idx] = float(len(X_cls)) / float(n_samples)

        # 2. Compute Centered Residuals
        # To estimate the common covariance matrix (pooled within-class covariance),
        # we subtract the corresponding class mean from each sample.
        # R = X - \mu_{y}
        X_centered = np.empty_like(X, dtype=NUMERIC_DTYPE)
        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_centered[mask] = X[mask] - self.means_[idx]

        # 3. Estimate Covariance using OAS
        # We use the centered residuals. OAS handles shrinkage analytically.
        oas = OAS(assume_centered=self.assume_centered)
        oas.fit(X_centered)

        self.covariance_ = oas.covariance_.astype(NUMERIC_DTYPE)
        self.precision_ = oas.precision_.astype(NUMERIC_DTYPE)

        # 4. Derive Linear Weights and Bias
        # W_k = P \mu_k
        # Shape: (n_classes, n_features) = (n_features, n_features) @ (n_features, n_classes)
        # Note: self.means_ is (n_classes, n_features), so we transpose it.
        self.coef_ = np.dot(self.means_, self.precision_)

        # b_k = -0.5 * (W_k . \mu_k) + log(\pi_k)
        # We compute the dot product row-wise.
        # (n_classes, n_features) * (n_classes, n_features) -> sum over axis 1
        term1 = -0.5 * np.sum(self.coef_ * self.means_, axis=1)
        term2 = np.log(self.priors_)
        self.intercept_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities, shape (n_samples, n_classes).
        """
        X = X.astype(NUMERIC_DTYPE, copy=False)

        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        # Linear Inference: Z = X W^T + b
        # X: (n_samples, n_features)
        # coef_: (n_classes, n_features) -> transpose -> (n_features, n_classes)
        # intercept_: (n_classes,)
        logits = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax to get probabilities
        # axis=1 ensures softmax is applied per sample across classes
        proba = softmax(logits, axis=1)

        return proba.astype(NUMERIC_DTYPE)

    def predict(self, X):
        """
        Predict class labels for X.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        """
        Returns the mean log loss on the given test data and labels.
        Note: Standard sklearn classifiers return accuracy for score(),
        but for this competition, log_loss is the primary metric.
        However, to maintain compatibility with sklearn tools (GridSearch, etc.),
        score usually returns accuracy. We will return negative log loss here
        to align with 'higher is better', or just return log loss and handle signs externally.

        For clarity in this specific pipeline, we return the Log Loss directly.
        """
        proba = self.predict_proba(X)
        return log_loss(y, proba, labels=self.classes_)
