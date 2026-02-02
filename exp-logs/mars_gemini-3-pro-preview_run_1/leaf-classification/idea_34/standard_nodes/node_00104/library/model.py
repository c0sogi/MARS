import os
import numpy as np
import pandas as pd
from scipy import linalg, special
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import library.config as config
import library.data_loader as data_loader
import library.preprocessing as preprocessing


class OASLinearDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using OAS covariance estimation.

    This classifier utilizes the SVD-based precision matrix from the OAS estimator
    to compute the linear discriminant weights. This approach is numerically
    more stable in high-dimensional, ill-conditioned regimes than explicit
    linear solvers (Cite solution_lesson_node_00062). It operates strictly in float64.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.covariance_ = None
        self.coef_ = None  # Weight matrix W
        self.intercept_ = None  # Bias vector b
        self.priors_ = None
        self.le_ = None

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Args:
            X (array-like): Training features of shape (N, D).
            y (array-like): Training labels of shape (N,).

        Returns:
            self
        """
        # Enforce strict float64 precision
        X = np.asarray(X, dtype=config.DTYPE)

        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # 1. Compute Empirical Class Means and Priors
        self.means_ = np.zeros((n_classes, n_features), dtype=config.DTYPE)
        self.priors_ = np.zeros(n_classes, dtype=config.DTYPE)

        for k in range(n_classes):
            mask = y_encoded == k
            X_k = X[mask]
            self.means_[k] = np.mean(X_k, axis=0)
            self.priors_[k] = len(X_k) / n_samples

        # 2. Compute Centered Residuals
        # R = X - mu_y
        means_expanded = self.means_[y_encoded]
        residuals = X - means_expanded

        # 3. Estimate Covariance using OAS
        # We use assume_centered=True because we have manually centered the residuals.
        oas = OAS(assume_centered=True)
        oas.fit(residuals)
        self.covariance_ = oas.covariance_.astype(config.DTYPE)

        # 4. Weight Derivation via SVD-based Precision Matrix
        # Cite solution_lesson_node_00062: Prefer library-provided precision_
        # W = mu * Sigma^-1
        self.coef_ = np.dot(self.means_, oas.precision_.astype(config.DTYPE))

        # 5. Compute Bias (Intercept)
        # b_k = -0.5 * (mu_k . W_k) + log(pi_k)
        quad_term = -0.5 * np.sum(self.means_ * self.coef_, axis=1)
        log_priors = np.log(self.priors_)
        self.intercept_ = quad_term + log_priors

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using the linear discriminant function.

        Args:
            X (array-like): Test features of shape (N, D).

        Returns:
            probs (array-like): Class probabilities of shape (N, K).
        """
        X = np.asarray(X, dtype=config.DTYPE)

        # Compute Logits (Linear Scores)
        # Z = X @ W^T + b
        scores = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax to convert logits to probabilities
        # scipy.special.softmax handles numerical stability
        probs = special.softmax(scores, axis=1)

        return probs


def train_and_predict():
    """
    Executes the full pipeline: Data Loading, Preprocessing, Training, Validation, and Submission.
    """
    print(
        "Starting Idea 34 Pipeline: Corrected Cholesky-Solved Exact-Precision OAS Discriminant"
    )

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading Data...")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, classes = (
        data_loader.load_data()
    )

    # -------------------------------------------------------------------------
    # 2. Preprocess Data
    # -------------------------------------------------------------------------
    print("\n[2/5] Preprocessing Data (Yeo-Johnson + StandardScaler)...")
    # Note: The preprocessor fits ONLY on X_train_raw to avoid leakage (Inductive Fit)
    X_train, X_val, X_test = preprocessing.preprocess_data(
        X_train_raw, X_val_raw, X_test_raw
    )

    # -------------------------------------------------------------------------
    # 3. Train Model
    # -------------------------------------------------------------------------
    print("\n[3/5] Training OASLinearDiscriminant...")
    model = OASLinearDiscriminant()
    model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 4. Validate
    # -------------------------------------------------------------------------
    print("\n[4/5] Validating...")
    val_probs = model.predict_proba(X_val)

    # Need to encode y_val to integers for log_loss
    y_val_enc = model.le_.transform(y_val)

    # Calculate Log Loss
    # We use the full precision probabilities
    loss = log_loss(y_val_enc, val_probs, labels=range(len(model.classes_)))
    print(f"Validation Multi-class Log Loss: {loss:.15f}")

    # -------------------------------------------------------------------------
    # 5. Generate Submission
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating Submission...")
    test_probs = model.predict_proba(X_test)

    # Ensure columns correspond to the sorted classes
    # model.classes_ comes from LabelEncoder, which sorts alphabetically.
    # data_loader.classes is also sorted. They should match.
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)
    submission_df.insert(0, "id", test_ids)

    # Save submission
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to: {config.SUBMISSION_FILE}")
