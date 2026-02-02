import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.covariance import OAS
from library import config


class AnalyticalOASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    A Pure Generative Model for High-Dimensional Classification using OAS-LDA.

    Learns a robust low-rank manifold using Linear Discriminant Analysis
    with Oracle Approximating Shrinkage (OAS) for covariance estimation.
    Uses the closed-form analytical solution for inference, avoiding
    numerical noise from discriminative calibration.

    Attributes:
        classes_ (np.ndarray): Unique class labels.
        W_ (np.ndarray): Projection matrix of shape (n_classes, n_features).
        b_init_ (np.ndarray): Analytical bias of shape (n_classes,).
    """

    def __init__(self):
        self.classes_ = None
        self.le_ = None
        self.W_ = None
        self.b_init_ = None
        self.means_ = None
        self.precision_ = None

    def fit(self, X, y):
        """
        Fits the model using the analytical OAS-LDA approach.

        Args:
            X (array-like): Training features of shape (N, D).
            y (array-like): Training labels of shape (N,).

        Returns:
            self
        """
        # 1. Data Preparation
        # Enforce float64 precision for numerical stability
        X = np.array(X, dtype=config.FLOAT_PRECISION)

        # Encode labels to integers 0..K-1
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # 2. Generative Phase: Manifold Learning via OAS-LDA
        # We need to compute means and center the data to estimate covariance
        self.means_ = np.zeros((n_classes, n_features), dtype=config.FLOAT_PRECISION)
        priors = np.zeros(n_classes, dtype=config.FLOAT_PRECISION)
        X_centered = np.zeros_like(X, dtype=config.FLOAT_PRECISION)

        for k in range(n_classes):
            # Boolean mask for class k
            mask = y_encoded == k
            X_k = X[mask]

            # Estimate priors and means
            priors[k] = len(X_k) / n_samples
            self.means_[k] = np.mean(X_k, axis=0)

            # Center the data for this class
            X_centered[mask] = X_k - self.means_[k]

        # Estimate Precision Matrix (Inverse Covariance) using OAS
        # assume_centered=True is critical as we have manually centered the data
        # based on class-specific means. Cite solution_lesson_node_00061
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)
        # Use precision_ directly. Cite solution_lesson_node_00062
        self.precision_ = oas.precision_.astype(config.FLOAT_PRECISION)

        # Compute Projection Matrix W = means @ precision
        # Shape: (K, D)
        self.W_ = self.means_ @ self.precision_

        # Compute Analytical Bias b_init based on Gaussian priors
        # b_init_k = -0.5 * (mu_k^T @ Sigma^-1 @ mu_k) + log(pi_k)
        #          = -0.5 * (mu_k^T @ W_k^T) + log(pi_k)
        # We use a vectorized dot product (sum of element-wise multiplication)
        # Cite solution_lesson_node_00055 (Linear Formulation)
        term1 = -0.5 * np.sum(self.means_ * self.W_, axis=1)
        term2 = np.log(priors)
        self.b_init_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for new data.

        Args:
            X (array-like): Features of shape (N, D).

        Returns:
            np.ndarray: Probabilities of shape (N, K).
        """
        # Enforce precision
        X = np.array(X, dtype=config.FLOAT_PRECISION)

        # 1. Projection & Inference
        # Compute logits analytically: L = X @ W.T + b
        # Cite solution_lesson_node_00077 (Avoid Discriminative Calibration)
        logits = X @ self.W_.T + self.b_init_

        # 2. Softmax (Numerically Stable)
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 4. Clipping
        # Ensure probabilities are within the safe range [1e-15, 1 - 1e-15]
        # as per the competition metric specification.
        probs = np.clip(probs, config.CLIP_EPSILON, 1.0 - config.CLIP_EPSILON)

        return probs

    def predict(self, X):
        """
        Predicts class labels for new data.

        Args:
            X (array-like): Features of shape (N, D).

        Returns:
            np.ndarray: Predicted class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]
