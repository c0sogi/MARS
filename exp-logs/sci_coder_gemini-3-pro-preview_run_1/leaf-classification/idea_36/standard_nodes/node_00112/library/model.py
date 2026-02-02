import numpy as np
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder

from library.config import FLOAT_PRECISION
from library.utils import stable_softmax


class OASLinearDiscriminant:
    """
    Linear Discriminant Analysis using Oracle Approximating Shrinkage (OAS).

    Uses analytical shrinkage estimation for the covariance matrix and
    computes the linear discriminant weights directly using the estimated precision.
    """

    def __init__(self):
        self.classes_ = None
        self.le_ = None
        self.W_ = None  # Weight matrix (n_classes, n_features)
        self.b_ = None  # Bias vector (n_classes,)

    def fit(self, X, y):
        """
        Fits the OAS-LDA model.
        """
        # Ensure float64
        X = X.astype(FLOAT_PRECISION)

        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Initialize stats
        means = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        priors = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        # Compute means and priors
        for k in range(n_classes):
            X_k = X[y_encoded == k]
            means[k] = np.mean(X_k, axis=0)
            priors[k] = len(X_k) / n_samples

        # Center data manually to ensure geometric consistency with means
        # Cite solution_lesson_node_00061
        X_centered = np.zeros_like(X, dtype=FLOAT_PRECISION)
        for k in range(n_classes):
            mask = y_encoded == k
            X_centered[mask] = X[mask] - means[k]

        # Estimate precision matrix using OAS
        # Cite solution_lesson_node_00108: Analytical shrinkage > Grid Search
        # Cite solution_lesson_node_00047: OAS > Ledoit-Wolf
        # Cite solution_lesson_node_00061: assume_centered=True
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)

        # Use library-provided precision matrix (SVD-based pseudo-inverse)
        # Cite solution_lesson_node_00062
        precision = oas.precision_

        # Compute Linear Discriminant Weights and Bias
        # Linear Formulation: Score = x.T @ W.T + b
        # Cite solution_lesson_node_00055

        # W = means @ precision (Shape: n_classes x n_features)
        self.W_ = np.dot(means, precision)

        # b = -0.5 * diag(means @ W.T) + log(priors)
        # Efficiently compute diagonal of (means @ W.T)
        term1 = -0.5 * np.sum(means * self.W_, axis=1)
        term2 = np.log(priors)
        self.b_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for samples in X.
        """
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X = X.astype(FLOAT_PRECISION)

        # Linear Inference: Z = X * W^T + b
        # Cite solution_lesson_node_00057: Maintain float64 for inference
        logits = np.dot(X, self.W_.T) + self.b_

        # Softmax
        probs = stable_softmax(logits)

        return probs
