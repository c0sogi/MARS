import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from library.config import FLOAT_PRECISION


class OASLinearDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using the Oracle Approximating Shrinkage (OAS)
    estimator for the covariance matrix.

    This implementation:
    1. Operates strictly in FLOAT_PRECISION (float64).
    2. Uses a shared covariance matrix assumption (Linear Discriminant).
    3. Estimates covariance using OAS on class-centered residuals.
    4. Pre-compiles decision boundaries (W, b) for fast, linear inference.
    """

    def __init__(self):
        self.classes_ = None
        self.le_ = None
        self.means_ = None
        self.priors_ = None
        self.precision_ = None
        self.W_ = None
        self.b_ = None

    def fit(self, X, y):
        """
        Fits the OAS-LDA model.

        Args:
            X (array-like): Training features of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,).

        Returns:
            self: Returns the instance itself.
        """
        # 1. Input Validation and Precision Casting
        X = np.array(X, dtype=FLOAT_PRECISION)
        y = np.array(y)

        n_samples, n_features = X.shape

        # 2. Encode Labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)

        # 3. Compute Empirical Means and Priors
        # We compute the arithmetic mean for each class.
        self.means_ = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        # We also need residuals for covariance estimation
        residuals = np.zeros_like(X, dtype=FLOAT_PRECISION)

        for k in range(n_classes):
            mask = y_encoded == k
            X_k = X[mask]

            if X_k.shape[0] == 0:
                raise ValueError(f"Class {self.classes_[k]} has no samples.")

            # Arithmetic Mean
            mean_k = np.mean(X_k, axis=0)
            self.means_[k] = mean_k

            # Prior
            self.priors_[k] = FLOAT_PRECISION(X_k.shape[0]) / FLOAT_PRECISION(n_samples)

            # Center data (Residuals)
            residuals[mask] = X_k - mean_k

        # 4. Estimate Covariance using OAS
        # We use assume_centered=True because we have manually centered the data
        # by subtracting class means (residuals).
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        # Extract the precision matrix (Inverse Covariance)
        # OAS implementation in sklearn provides this via SVD-based pseudo-inverse
        self.precision_ = oas.precision_.astype(FLOAT_PRECISION)

        # 5. Derive Linear Decision Boundaries (W, b)
        # Discriminant function: delta_k(x) = x.T @ (P @ mu_k) - 0.5 * mu_k.T @ P @ mu_k + log(pi_k)
        # Linear form: Z = X @ W.T + b

        # W matrix: Rows are (P @ mu_k).T
        # Shape: (n_classes, n_features)
        # Since P is symmetric, P @ mu_k is equivalent to (mu_k @ P).T
        self.W_ = np.dot(self.means_, self.precision_)

        # Bias vector b
        # Term 1: -0.5 * diag(mu_k.T @ P @ mu_k)
        # We can compute this efficiently using element-wise multiplication and sum
        term1 = -0.5 * np.sum(self.W_ * self.means_, axis=1)

        # Term 2: log(pi_k)
        term2 = np.log(self.priors_)

        self.b_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (array-like): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities of shape (n_samples, n_classes).
        """
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # 1. Cast Input
        X = np.array(X, dtype=FLOAT_PRECISION)

        # 2. Compute Logits (Linear Scoring)
        # Z = X @ W.T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # 3. Apply Softmax
        # scipy.special.softmax handles numerical stability (exp-normalize trick)
        probs = softmax(logits, axis=1)

        return probs.astype(FLOAT_PRECISION)

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X (array-like): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.le_.inverse_transform(indices)


def create_submission_file(model, X_test, ids_test, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (OASLinearDiscriminant): Fitted model.
        X_test (array-like): Test features.
        ids_test (array-like): Test image IDs.
        output_path (str): Path to save the submission CSV.
    """
    # Generate probabilities
    probs = model.predict_proba(X_test)

    # Create DataFrame
    # Columns must be 'id' followed by class names
    df_sub = pd.DataFrame(probs, columns=model.classes_)
    df_sub.insert(0, "id", ids_test)

    # Save to CSV
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
