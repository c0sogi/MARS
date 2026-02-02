import numpy as np
from sklearn.covariance import OAS
from library.config import NUMERIC_DTYPE, FEATURE_PREFIXES
from library.utils import calculate_log_loss, save_submission
from library.data_loader import load_and_process_data


class GlobalOASDiscriminant:
    """
    A Global Linear Discriminant Analysis model using OAS covariance estimation.
    Pools all features (Margin, Shape, Texture) to capture cross-group correlations.
    """

    def __init__(self):
        self.W = None
        self.b = None
        self.classes = None

    def fit(self, X_dict, y):
        """
        Fits the global model.

        Args:
            X_dict (dict): Dictionary containing feature arrays for each group.
            y (array): Target labels (integers).
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Concatenate features
        # Cite solution_lesson_node_00015: Prefer global models that pool data across all classes
        X_parts = [X_dict[group] for group in FEATURE_PREFIXES]
        X = np.hstack(X_parts).astype(NUMERIC_DTYPE)

        n_samples, n_features = X.shape

        # Calculate Priors
        counts = np.bincount(y)
        priors = counts / counts.sum()
        self.log_priors = np.log(priors + 1e-15).astype(NUMERIC_DTYPE)

        # Compute Means
        # Cite solution_lesson_node_00066: Arithmetic Mean is optimal for Gaussianized data
        means = np.zeros((n_classes, n_features), dtype=NUMERIC_DTYPE)
        for k in self.classes:
            means[k] = np.mean(X[y == k], axis=0)

        # Compute Residuals (Centered Data)
        X_centered = np.zeros_like(X)
        for k in self.classes:
            X_centered[y == k] = X[y == k] - means[k]

        # Estimate Precision Matrix via OAS
        # Cite solution_lesson_node_00047: OAS outperforms Ledoit-Wolf on Gaussianized Data
        # Cite solution_lesson_node_00061: Enforce geometric consistency with assume_centered=True
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)

        # Cite solution_lesson_node_00062: Use library-provided precision_ attribute
        precision = oas.precision_.astype(NUMERIC_DTYPE)

        # Compute Linear Discriminant Parameters
        # Cite solution_lesson_node_00055: Use Linear Formulation
        self.W = np.dot(means, precision)

        # Quadratic Bias term
        quad_term = -0.5 * np.sum(means * self.W, axis=1)
        self.b = quad_term + self.log_priors

    def predict_proba(self, X_dict):
        """
        Predicts class probabilities using the global discriminant.
        """
        # Concatenate features
        X_parts = [X_dict[group] for group in FEATURE_PREFIXES]
        X = np.hstack(X_parts).astype(NUMERIC_DTYPE)

        # Compute logits
        # Cite solution_lesson_node_00055: Linear Formulation
        logits = np.dot(X, self.W.T) + self.b

        # Softmax in high precision
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs
