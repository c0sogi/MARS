import numpy as np
from sklearn.covariance import OAS
from library.config import NUMERIC_DTYPE, FEATURE_PREFIXES
from library.utils import calculate_log_loss, save_submission
from library.data_loader import load_and_process_data


class GlobalOASDiscriminant:
    """
    A Global Linear Discriminant Analysis model using OAS covariance estimation.
    It concatenates all feature groups into a single vector to capture
    inter-group correlations, avoiding the independence assumption of factorized models.
    """

    def __init__(self):
        self.W = None
        self.b = None
        self.classes = None

    def fit(self, X_dict, y):
        """
        Fits the global model.
        Cite Lesson 00015: Avoid "Divide and Conquer" architectures on small, high-cardinality datasets.
        Cite Lesson 00008: Single LDA model is sufficient.
        """
        # Concatenate all feature groups
        X_parts = [X_dict[group] for group in FEATURE_PREFIXES]
        X = np.hstack(X_parts).astype(NUMERIC_DTYPE)

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        # 1. Calculate Priors
        counts = np.bincount(y)
        priors = counts / counts.sum()
        log_priors = np.log(priors + 1e-15).astype(NUMERIC_DTYPE)

        # 2. Compute Class Means
        means = np.zeros((n_classes, n_features), dtype=NUMERIC_DTYPE)
        for k in self.classes:
            means[k] = np.mean(X[y == k], axis=0)

        # 3. Center Data (Compute Residuals)
        X_centered = np.zeros_like(X)
        for k in self.classes:
            X_centered[y == k] = X[y == k] - means[k]

        # 4. Estimate Precision Matrix via OAS
        # Cite Lesson 00047: OAS outperforms Ledoit-Wolf on Gaussianized Data
        # Cite Lesson 00061: Enforce geometric consistency (assume_centered=True)
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)
        # Cite Lesson 00062: Use precision_ attribute
        precision = oas.precision_.astype(NUMERIC_DTYPE)

        # 5. Compute Linear Discriminant Parameters
        # Cite Lesson 00055: Use Linear Formulation
        self.W = np.dot(means, precision)

        # Bias: -0.5 * diag(means @ precision @ means.T) + log_prior
        quad_term = -0.5 * np.sum(means * self.W, axis=1)
        self.b = quad_term + log_priors

    def predict_proba(self, X_dict):
        # Concatenate all feature groups
        X_parts = [X_dict[group] for group in FEATURE_PREFIXES]
        X = np.hstack(X_parts).astype(NUMERIC_DTYPE)

        # Linear Score: X @ W.T + b
        # Cite Lesson 00057: Inference precision (float64)
        logits = np.dot(X, self.W.T) + self.b

        # Softmax
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs
