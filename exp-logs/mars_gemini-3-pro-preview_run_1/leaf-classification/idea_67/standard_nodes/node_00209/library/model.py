import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from library.config import FLOAT_PRECISION, OAS_ASSUME_CENTERED
from library.utils import set_seed


class OASLinearDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using an OAS Covariance Backbone.

    Implements the 'Sanitized Axis-Augmented High-Precision OAS Discriminant' strategy.
    It estimates the covariance matrix using Oracle Approximating Shrinkage (OAS) on
    centered residuals and pre-compiles linear decision boundaries for exact analytical inference.

    Attributes:
        classes_ (np.ndarray): Unique class labels.
        W_ (np.ndarray): Weight matrix of shape (n_classes, n_features).
        b_ (np.ndarray): Bias vector of shape (n_classes,).
        precision_ (np.ndarray): Estimated precision matrix of shape (n_features, n_features).
    """

    def __init__(self):
        """
        Initialize the OASLinearDiscriminant.
        Parameters are drawn strictly from library.config.
        """
        self.assume_centered = OAS_ASSUME_CENTERED
        self.dtype = FLOAT_PRECISION

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Steps:
        1. Compute empirical class means and priors.
        2. Compute residuals (X - class_means).
        3. Estimate covariance/precision using OAS on residuals.
        4. Derive linear decision boundaries (W, b).

        Args:
            X (array-like): Training data of shape (n_samples, n_features).
            y (array-like): Target values of shape (n_samples,).

        Returns:
            self: Returns the instance itself.
        """
        # Ensure reproducibility
        set_seed()

        # Input validation and conversion to high precision
        X, y = check_X_y(X, y, dtype=self.dtype)

        # Encode labels
        self.le_ = LabelEncoder()
        y_idx = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Parameter Estimation (Means and Priors)
        means = np.zeros((n_classes, n_features), dtype=self.dtype)
        priors = np.zeros(n_classes, dtype=self.dtype)

        # Efficiently compute means and counts
        for k in range(n_classes):
            X_k = X[y_idx == k]
            means[k] = np.mean(X_k, axis=0)
            priors[k] = len(X_k) / len(X)

        # 2. Compute Residuals
        # R = X - mu_y
        # We map the means back to the size of X to subtract efficiently
        residuals = X - means[y_idx]

        # 3. Estimate Covariance/Precision using OAS
        # We use assume_centered=True because we manually centered the data via residuals
        oas = OAS(assume_centered=self.assume_centered)
        oas.fit(residuals)

        # Extract the precision matrix (inverse covariance)
        # OAS.precision_ is calculated via SVD/pseudo-inverse internally in sklearn
        self.precision_ = oas.precision_.astype(self.dtype)

        # 4. Weight Derivation (Linear Formulation)
        # W_k = P * mu_k
        # Since P is symmetric, W = (P @ means.T).T = means @ P
        # Shape: (n_classes, n_features)
        self.W_ = np.dot(means, self.precision_)

        # Bias Derivation
        # b_k = -0.5 * (mu_k^T * W_k) + log(pi_k)
        # We compute the quadratic term efficiently
        # np.einsum('ij,ij->i', A, B) computes the dot product for each row
        quadratic_term = -0.5 * np.einsum("ij,ij->i", means, self.W_)
        log_priors = np.log(priors)

        self.b_ = quadratic_term + log_priors

        return self

    def predict_proba(self, X):
        """
        Estimate probability.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probability of the sample for each class in the model,
                        shape (n_samples, n_classes).
        """
        check_is_fitted(self)
        X = check_array(X, dtype=self.dtype)

        # Linear Scoring: Z = X @ W.T + b
        # X: (N, D), W_: (K, D), b_: (K,)
        # Result: (N, K)
        logits = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax
        # Using scipy.special.softmax for numerical stability
        probs = softmax(logits, axis=1)

        return probs

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]
