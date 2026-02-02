import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
import library.config as config


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    A custom Linear Discriminant Classifier that uses the Oracle Approximating Shrinkage (OAS)
    estimator for the covariance matrix. This implementation is designed for high-precision
    inference (float64) and algebraic stability.

    It solves the discriminant function:
        delta_k(x) = x^T P mu_k - 0.5 * mu_k^T P mu_k + log(pi_k)

    Where:
        P: Precision matrix (inverse covariance) estimated via OAS.
        mu_k: Mean vector for class k.
        pi_k: Prior probability for class k.

    This is algebraically converted to a linear decision boundary:
        Z = X W^T + b
    """

    def __init__(self, assume_centered=True):
        """
        Args:
            assume_centered (bool): Parameter for the OAS estimator.
                                    Defaults to True as we manually center data based on class means.
        """
        self.assume_centered = assume_centered

    def fit(self, X, y):
        """
        Fits the model using the OAS covariance estimator.

        1. Computes class means and priors.
        2. Centers data (residuals).
        3. Fits OAS on residuals to get Precision matrix.
        4. Derives weights (W) and bias (b).

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels (integer encoded).

        Returns:
            self
        """
        # Enforce high precision
        X = check_array(X, dtype=config.FLOAT_PRECISION)
        X, y = check_X_y(X, y, dtype=config.FLOAT_PRECISION)
        # y must be integers for indexing, though check_X_y might cast to float if X is float.
        # We ensure y is int for processing.
        y = y.astype(int)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # 1. Compute Empirical Means and Priors
        # Matrix M: (n_classes, n_features)
        self.means_ = np.zeros((n_classes, n_features), dtype=config.FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=config.FLOAT_PRECISION)

        for idx, cls in enumerate(self.classes_):
            X_k = X[y == cls]
            self.means_[idx] = np.mean(X_k, axis=0)
            self.priors_[idx] = len(X_k) / n_samples

        # 2. Compute Residuals (Centering)
        # R = X - mu_y
        # We subtract the corresponding class mean from each sample
        X_centered = X - self.means_[y]

        # 3. Estimate Covariance / Precision using OAS
        # We use assume_centered=True because we have already centered X around class means.
        # This corresponds to the "within-class scatter" assumption of LDA.
        self.oas_ = OAS(assume_centered=self.assume_centered)
        self.oas_.fit(X_centered)

        # P: Precision Matrix (n_features, n_features)
        self.precision_ = self.oas_.precision_.astype(config.FLOAT_PRECISION)

        # 4. Derive Linear Weights and Bias
        # Term 1: x^T P mu_k  ->  Linear Weight W_k = P mu_k
        # Since P is symmetric, W = M P
        # W shape: (n_classes, n_features)
        self.coef_ = np.dot(self.means_, self.precision_)

        # Term 2: -0.5 * mu_k^T P mu_k + log(pi_k)
        # We can compute mu_k^T P mu_k efficiently as row-wise dot product of M and W
        # (since W_k = P mu_k)
        term2 = -0.5 * np.sum(self.means_ * self.coef_, axis=1)
        term3 = np.log(self.priors_)

        self.intercept_ = term2 + term3

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the pre-computed linear boundaries.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes).
        """
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X, dtype=config.FLOAT_PRECISION)

        # Compute Logits: Z = X W^T + b
        # X: (N, D), W^T: (D, K), b: (K,) -> Z: (N, K)
        logits = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax
        # Using scipy.special.softmax is numerically stable
        proba = softmax(logits, axis=1)

        return proba

    def predict(self, X):
        """
        Predicts class labels.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.classes_[indices]
