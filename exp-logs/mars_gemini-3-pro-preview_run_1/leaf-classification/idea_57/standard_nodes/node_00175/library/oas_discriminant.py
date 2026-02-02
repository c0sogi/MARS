import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from scipy.special import softmax


class OASLinearDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using OAS Covariance Estimator.

    This model implements the 'Sanitized Parsimonious Integral-Geometric High-Precision'
    strategy by performing exact analytical inference using float64 precision.
    It fits an OAS estimator on the class-conditional residuals to derive
    robust linear decision boundaries.
    """

    def __init__(self):
        """
        Initialize the OASLinearDiscriminant.
        """
        self.classes_ = None
        self.means_ = None
        self.precision_ = None
        self.weights_ = None
        self.bias_ = None
        self.estimator = None

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Args:
            X (array-like): Training vector, where n_samples is the number of samples
                            and n_features is the number of features.
            y (array-like): Target vector relative to X.

        Returns:
            self: Object.
        """
        # Enforce float64 precision for input data
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # Identify classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize parameter containers
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)

        # 1. Parameter Estimation
        # Compute empirical class means and priors
        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.means_[idx] = np.mean(X_cls, axis=0)
            priors[idx] = float(len(X_cls)) / len(X)

        # Compute Residuals (Centered Data)
        # We subtract the specific class mean from each sample to compute the
        # pooled within-class scatter/covariance.
        X_centered = np.empty_like(X, dtype=np.float64)
        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_centered[mask] = X[mask] - self.means_[idx]

        # Fit OAS Estimator on residuals
        # assume_centered=True is used because X_centered already has zero mean per class logic
        self.estimator = OAS(assume_centered=True)
        self.estimator.fit(X_centered)

        # Extract Precision Matrix (Inverse Covariance)
        # We use the SVD-based pseudo-inverse provided by the OAS implementation
        self.precision_ = self.estimator.precision_

        # 2. Weight Derivation (Linear Formulation)
        # Weights_k = Precision @ Mean_k
        # Shape: (n_classes, n_features)
        self.weights_ = self.means_ @ self.precision_

        # Bias Derivation
        # b_k = -0.5 * (W_k . mu_k) + log(pi_k)
        # The quadratic term cancels out the distance metric instability
        quad_term = -0.5 * np.sum(self.weights_ * self.means_, axis=1)
        self.bias_ = quad_term + np.log(priors)

        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.

        Args:
            X (array-like): Input data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Returns the probability of the sample for each class in the model,
                        where classes are ordered as they are in self.classes_.
        """
        # Enforce float64 precision
        X = np.asarray(X, dtype=np.float64)

        # Linear Scoring: Z = X @ W^T + b
        # X: (N, D), W: (K, D), b: (K,) -> logits: (N, K)
        logits = X @ self.weights_.T + self.bias_

        # Apply Softmax in float64
        probs = softmax(logits, axis=1)

        # Clip probabilities to avoid log(0) issues in metrics
        # Range: [1e-15, 1 - 1e-15]
        eps = 1e-15
        probs = np.clip(probs, eps, 1.0 - eps)

        return probs

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (array-like): Input data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]
