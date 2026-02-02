import numpy as np
import pandas as pd
import os
from sklearn.covariance import OAS
from scipy.special import softmax

from library.config import SUBMISSION_FILE, SUBMISSION_DIR
from library.utils import compute_log_loss
from library.data import DataManager


class OASDiscriminant:
    """
    A custom Linear Discriminant Classifier that uses the Oracle Approximating Shrinkage (OAS)
    estimator to compute the precision matrix (inverse covariance) from centered residuals.

    This implementation explicitly formulates the decision boundaries using linear algebra
    in float64 precision to avoid the numerical instability of distance-based formulations.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.precision_ = None
        self.W_ = None
        self.b_ = None

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features).
            y (np.ndarray): Training labels, shape (n_samples,).
        """
        # Ensure float64
        X = X.astype(np.float64)

        # Identify classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize statistics
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        # Compute empirical means and priors
        for idx, k in enumerate(self.classes_):
            X_k = X[y == k]
            self.means_[idx] = np.mean(X_k, axis=0)
            self.priors_[idx] = len(X_k) / len(X)

        # Compute centered residuals (X - mu_y)
        # This removes the class mean from each sample to isolate the noise covariance
        residuals = np.zeros_like(X, dtype=np.float64)
        for idx, k in enumerate(self.classes_):
            mask = y == k
            residuals[mask] = X[mask] - self.means_[idx]

        # Estimate Covariance/Precision using OAS
        # assume_centered=True because we manually centered the data above
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        self.precision_ = oas.precision_

        # Pre-compute Linear Decision Boundary Parameters
        # Discriminant function: delta_k(x) = x.T @ P @ mu_k - 0.5 * mu_k.T @ P @ mu_k + log(pi_k)
        # Linear form: Z = X @ W.T + b

        # W = M @ P (Shape: n_classes x n_features)
        # Note: In the formula x.T @ P @ mu_k, P is symmetric.
        # So (P @ mu_k).T = mu_k.T @ P.
        self.W_ = np.dot(self.means_, self.precision_)

        # Quadratic term: 0.5 * diag(mu_k.T @ P @ mu_k)
        # We can compute this efficiently as row-wise dot product of Means and W
        quad_term = 0.5 * np.sum(self.means_ * self.W_, axis=1)

        # Bias b = -0.5 * mu^T P mu + log(prior)
        self.b_ = -quad_term + np.log(self.priors_)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for samples in X.

        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities, shape (n_samples, n_classes).
        """
        X = X.astype(np.float64)

        # Linear Score: Z = X @ W.T + b
        # Shapes: (N, F) @ (K, F).T + (K,) -> (N, K)
        logits = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax to get probabilities
        probs = softmax(logits, axis=1)

        return probs


def train_and_evaluate(load_cached_data=True):
    """
    Orchestrates the data loading, training, validation, and submission generation.

    Args:
        load_cached_data (bool): Whether to load pre-computed features/preprocessing from cache.
    """
    # 1. Load Data using the centralized DataManager
    # This handles geometric feature extraction, merging, and high-precision preprocessing
    dm = DataManager()
    data = dm.get_processed_data(load_cached_data=load_cached_data)

    X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test, classes = data

    print(
        f"Data Loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # 2. Initialize and Train Model
    print("Training OAS Discriminant...")
    model = OASDiscriminant()
    model.fit(X_train, y_train)

    # 3. Validation
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Compute metric using the competition-specific log loss function
    # (handles clipping and rescaling internally)
    val_loss = compute_log_loss(y_val, val_probs)
    print(f"Validation Multi-class Log Loss: {val_loss:.15f}")

    # 4. Generate Submission
    print("Generating predictions for Test Set...")
    test_probs = model.predict_proba(X_test)

    # Create submission DataFrame
    df_sub = pd.DataFrame(test_probs, columns=classes)
    df_sub.insert(0, "id", ids_test)

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Save
    df_sub.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")
