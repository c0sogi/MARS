import numpy as np
import pandas as pd
from scipy import linalg, special
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
import os

from library import config
from library import metrics
from library import data_loader
from library import preprocessing


class CholeskyOASLinearDiscriminant:
    """
    A Linear Discriminant Classifier that uses the Oracle Approximating Shrinkage (OAS)
    estimator for covariance and solves for weights using Cholesky decomposition.

    This implementation strictly adheres to float64 precision and avoids explicit
    matrix inversion to maximize numerical stability and decision boundary resolution.
    """

    def __init__(self):
        self.classes_ = None
        self.W_ = None
        self.b_ = None
        self.le_ = None
        self.covariance_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation and Cholesky-based solver.

        Args:
            X (array-like): Training features (n_samples, n_features).
            y (array-like): Target labels (n_samples,).

        Returns:
            self
        """
        # Enforce Double Precision
        X = np.array(X, dtype=np.float64)

        # Encode labels to 0..K-1
        self.le_ = LabelEncoder()
        y_idx = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Empirical Class Means and Priors
        # We use the arithmetic mean which is the MLE for Gaussian data.
        means = np.zeros((n_classes, n_features), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)

        for k in range(n_classes):
            mask = y_idx == k
            X_k = X[mask]
            means[k] = np.mean(X_k, axis=0)
            priors[k] = len(X_k) / len(X)

        # 2. Compute Residuals
        # Center the data by subtracting the corresponding class mean.
        # R = X - mu_y
        residuals = X - means[y_idx]

        # 3. Estimate Covariance Matrix via OAS
        # We use assume_centered=True because we have explicitly centered the data (residuals).
        # This ensures the estimator uses the correct degrees of freedom and geometric assumptions.
        oas = OAS(assume_centered=True)
        oas.fit(residuals)
        self.covariance_ = oas.covariance_

        # 4. Exact Weight Derivation via Cholesky Solver
        # We solve the system: Sigma * W^T = mu^T
        # Instead of W = Sigma^-1 * mu, we use Cholesky to avoid explicit inversion.

        # Decompose Sigma = L * L^T
        c, lower = linalg.cho_factor(self.covariance_, lower=True)

        # Solve for W^T (shape: n_features, n_classes)
        # cho_solve uses the factorization to solve Ax=b efficiently and stably.
        W_trans = linalg.cho_solve((c, lower), means.T)

        # Store W (shape: n_classes, n_features)
        self.W_ = W_trans.T

        # 5. Compute Bias Term
        # b_k = -0.5 * (W_k . mu_k) + log(pi_k)
        # The quadratic term represents the Mahalanobis distance correction.
        # We perform row-wise dot product between W and means.
        term_quadratic = -0.5 * np.sum(self.W_ * means, axis=1)
        term_logprior = np.log(priors)
        self.b_ = term_quadratic + term_logprior

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linear decision function and softmax.

        Args:
            X (array-like): Input features.

        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes).
        """
        X = np.array(X, dtype=np.float64)

        # Linear Decision Function: Z = X * W^T + b
        # Dimensions: (N, D) * (D, K) + (K,) -> (N, K)
        Z = X @ self.W_.T + self.b_

        # Apply Softmax to convert logits to probabilities
        # scipy.special.softmax is numerically stable
        return special.softmax(Z, axis=1)


def train_and_predict():
    """
    Orchestrates the full pipeline: data loading, preprocessing, model training,
    evaluation, and submission generation.
    """
    print("Initializing Exact Cholesky-OAS Pipeline...")

    # 1. Load Data
    # The data loader handles caching and ensures float64 precision.
    (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids) = (
        data_loader.load_datasets(load_cached_data=True)
    )

    # 2. Preprocess Data
    # Apply Yeo-Johnson and Standard Scaling (fitted on train only).
    X_train_trans, X_val_trans, X_test_trans = preprocessing.preprocess_datasets(
        X_train, X_val, X_test, load_cached_data=True
    )

    # 3. Train Model
    print("Training CholeskyOASLinearDiscriminant...")
    model = CholeskyOASLinearDiscriminant()
    model.fit(X_train_trans, y_train)

    # 4. Evaluate on Validation Set
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val_trans)

    # Calculate Log Loss using the competition metric definition
    score = metrics.calculate_log_loss(y_val, val_probs, labels=model.classes_)
    print(f"Validation Log Loss: {score}")

    # 5. Generate Submission
    print("Generating predictions for Test Set...")
    test_probs = model.predict_proba(X_test_trans)

    # Create submission DataFrame
    # Columns must be the class names (sorted alphabetically, which LabelEncoder does)
    submission = pd.DataFrame(test_probs, columns=model.classes_)

    # Insert 'id' column at the beginning
    submission.insert(0, "id", test_ids)

    # Save to disk
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
