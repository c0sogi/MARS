import numpy as np
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from library.config import Config


class OASLinearDiscriminant:
    """
    Custom Linear Discriminant Classifier using OAS Covariance Estimation.
    Implemented for high-precision float64 inference.

    This model assumes a shared covariance matrix across all classes (Linear Discriminant)
    but uses the Oracle Approximating Shrinkage (OAS) estimator to robustly estimate
    covariance in high-dimensional, potentially collinear feature spaces.

    Attributes:
        classes_ (np.ndarray): Unique class labels.
        means_ (np.ndarray): Class means (centroids), shape (n_classes, n_features).
        precision_ (np.ndarray): Inverse covariance matrix, shape (n_features, n_features).
        priors_ (np.ndarray): Class prior probabilities, shape (n_classes,).
        W_ (np.ndarray): Weight matrix for linear decision boundary, shape (n_classes, n_features).
        b_ (np.ndarray): Bias vector for linear decision boundary, shape (n_classes,).
        le (LabelEncoder): Encoder for target labels.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.precision_ = None
        self.priors_ = None
        self.W_ = None
        self.b_ = None
        self.le = LabelEncoder()

    def fit(self, X, y):
        """
        Fits the OAS Linear Discriminant model.

        1. Computes empirical means and priors.
        2. Centers the data (residuals).
        3. Estimates covariance using OAS.
        4. Derives linear weights (W) and bias (b).

        Args:
            X (array-like): Training features, shape (n_samples, n_features).
            y (array-like): Target labels, shape (n_samples,).

        Returns:
            self: The fitted instance.
        """
        # Ensure input is float64
        X = np.array(X, dtype=Config.FLOAT_PRECISION)

        # Encode classes
        y_enc = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize statistics containers
        self.priors_ = np.zeros(n_classes, dtype=Config.FLOAT_PRECISION)
        self.means_ = np.zeros((n_classes, n_features), dtype=Config.FLOAT_PRECISION)

        # Compute Priors and Means per class
        for k in range(n_classes):
            # Boolean indexing for class k
            mask = y_enc == k
            X_k = X[mask]

            # Empirical Prior
            self.priors_[k] = len(X_k) / len(X)

            # Empirical Mean
            self.means_[k] = np.mean(X_k, axis=0)

        # Center data (Global covariance assumption)
        # Residuals R = X - mu_y
        # We construct R to fit the covariance matrix on the pooled within-class scatter
        R = np.zeros_like(X, dtype=Config.FLOAT_PRECISION)
        for k in range(n_classes):
            mask = y_enc == k
            R[mask] = X[mask] - self.means_[k]

        # Estimate Covariance using OAS
        # OAS is robust to high-dimensionality and collinearity.
        # assume_centered=True because we explicitly centered R above.
        oas = OAS(assume_centered=True)
        oas.fit(R)

        # Extract Precision Matrix (Inverse Covariance)
        self.precision_ = oas.precision_.astype(Config.FLOAT_PRECISION)

        # Pre-compute Linear Decision Boundaries
        # The discriminant function is:
        # delta_k(x) = x.T @ (Sigma^-1 @ mu_k) - 0.5 * (mu_k.T @ Sigma^-1 @ mu_k) + log(pi_k)

        # Term 1: Weights W
        # W_k = Sigma^-1 @ mu_k
        # We store W_ such that Z = X @ W_.T
        # Therefore, W_ should be (n_classes, n_features) where row k is (Sigma^-1 @ mu_k).T
        # W_ = means_ @ precision_ (since precision is symmetric)
        self.W_ = self.means_ @ self.precision_

        # Term 2 & 3: Bias b
        # b_k = -0.5 * (mu_k.T @ Sigma^-1 @ mu_k) + log(pi_k)
        self.b_ = np.zeros(n_classes, dtype=Config.FLOAT_PRECISION)
        for k in range(n_classes):
            # Quadratic form: mu_k @ P @ mu_k.T
            # Note: self.means_[k] is 1D array (n_features,), so @ works as dot product
            term_quad = -0.5 * (self.means_[k] @ self.precision_ @ self.means_[k])
            term_log_prior = np.log(self.priors_[k])
            self.b_[k] = term_quad + term_log_prior

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for samples in X.

        Args:
            X (array-like): Input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities, shape (n_samples, n_classes).
        """
        # Ensure input is float64
        X = np.array(X, dtype=Config.FLOAT_PRECISION)

        # Linear Discriminant Function: Z = X @ W.T + b
        # X: (N, F)
        # W_: (K, F) -> W_.T: (F, K)
        # b_: (K,)
        # Z: (N, K)
        Z = X @ self.W_.T + self.b_

        # Apply Softmax to get probabilities
        # axis=1 computes softmax across classes for each sample
        return softmax(Z, axis=1)
