import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from library import utils


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS).

    Implements the 'Sanitized Integral-Geometric High-Precision OAS Discriminant' logic:
    1. Fits OAS on class-centered residuals to estimate the precision matrix.
    2. Computes linear decision boundaries (Weights W, Bias b) analytically.
    3. Performs inference using linear projection and softmax in float64.
    """

    def __init__(self):
        self.classes_ = None
        self.le_ = None
        self.W_ = None  # Weights: (n_classes, n_features)
        self.b_ = None  # Bias: (n_classes,)
        self.precision_ = None
        self.means_ = None
        self.priors_ = None

    def fit(self, X, y):
        """
        Fits the OAS Discriminant model.

        Args:
            X (array-like): Feature matrix (n_samples, n_features).
            y (array-like): Target labels (n_samples,).

        Returns:
            self
        """
        # 1. Enforce Float64 Precision
        X = utils.enforce_float64(np.array(X))
        y = np.array(y)

        # 2. Encode Labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 3. Compute Empirical Class Means and Priors
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        # We need residuals for OAS: R = X - mu_y
        X_centered = np.zeros_like(X, dtype=np.float64)

        for k in range(n_classes):
            mask = y_encoded == k
            X_k = X[mask]

            # Prior: Empirical frequency
            self.priors_[k] = len(X_k) / len(X)

            # Mean: Arithmetic mean
            if len(X_k) > 0:
                mu_k = np.mean(X_k, axis=0)
                self.means_[k] = mu_k

                # Centering
                X_centered[mask] = X_k - mu_k
            else:
                # Handle rare case of missing class in split
                self.means_[k] = 0.0

        # 4. Estimate Covariance/Precision using OAS
        # assume_centered=True because we manually centered X using class means.
        # This ensures geometric consistency.
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)

        self.precision_ = utils.enforce_float64(oas.precision_)

        # 5. Derive Linear Decision Boundaries
        # Weights W_k = mu_k * P (Row vector formulation)
        # Shape: (n_classes, n_features)
        self.W_ = np.dot(self.means_, self.precision_)

        # Bias b_k = -0.5 * (W_k . mu_k) + log(pi_k)
        # Element-wise multiplication followed by sum over features gives dot product for each class
        quad_term = -0.5 * np.sum(self.W_ * self.means_, axis=1)

        # Add epsilon to priors to avoid log(0) in extreme edge cases, though priors > 0 by def.
        log_priors = np.log(self.priors_ + 1e-15)
        self.b_ = quad_term + log_priors

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the pre-compiled linear boundaries.

        Args:
            X (array-like): Feature matrix.

        Returns:
            array-like: Probabilities (n_samples, n_classes).
        """
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X = utils.enforce_float64(np.array(X))

        # 1. Linear Projection: Z = X W^T + b
        # X: (N, F), W_: (K, F) -> W_.T: (F, K)
        # Z: (N, K)
        logits = np.dot(X, self.W_.T) + self.b_

        # 2. Softmax (High Precision)
        # Applies softmax along the class axis
        probas = softmax(logits, axis=1)

        return probas

    def predict(self, X):
        """
        Predicts class labels.

        Args:
            X (array-like): Feature matrix.

        Returns:
            array-like: Predicted class labels.
        """
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.classes_[indices]
