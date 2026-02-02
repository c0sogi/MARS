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


class CholeskyOASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using OAS covariance estimation
    and exact Cholesky decomposition solver.

    This classifier explicitly solves the linear system Sigma * W = Mu using
    Cholesky factorization to avoid the spectral truncation inherent in SVD-based
    pseudo-inverses. It operates strictly in float64.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.covariance_ = None
        self.precision_chol_ = None  # Stores tuple (c, lower) for cho_solve
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
        # This centers the data around the class means to estimate the
        # common within-class covariance matrix.
        means_expanded = self.means_[y_encoded]
        residuals = X - means_expanded

        # 3. Estimate Covariance using OAS
        # We use assume_centered=True because we have manually centered the residuals.
        # OAS shrinkage guarantees the matrix is Positive Definite (PD).
        oas = OAS(assume_centered=True)
        oas.fit(residuals)
        self.covariance_ = oas.covariance_.astype(config.DTYPE)

        # 4. Exact Weight Derivation via Cholesky Decomposition
        # LDA Discriminant Function: g_k(x) = x^T * Sigma^-1 * mu_k - 0.5 * mu_k^T * Sigma^-1 * mu_k + log(pi_k)
        # Linear weights W = mu * Sigma^-1  =>  W * Sigma = mu  =>  Sigma * W^T = mu^T
        # We solve for W^T (shape D, K)

        try:
            # Compute Cholesky factorization Sigma = L * L.T
            # check_finite=False is safe as we control the input pipeline
            c, lower = linalg.cho_factor(
                self.covariance_, lower=True, check_finite=False
            )
            self.precision_chol_ = (c, lower)
        except linalg.LinAlgError:
            print("Warning: Cholesky factorization failed. Adding jitter.")
            # Fallback for numerical stability (rare with OAS)
            jitter = np.eye(n_features, dtype=config.DTYPE) * 1e-8
            self.covariance_ += jitter
            c, lower = linalg.cho_factor(
                self.covariance_, lower=True, check_finite=False
            )
            self.precision_chol_ = (c, lower)

        # Solve A * X = B  =>  Sigma * W^T = Means^T
        W_t = linalg.cho_solve(self.precision_chol_, self.means_.T, check_finite=False)
        self.coef_ = W_t.T  # Shape (K, D)

        # 5. Compute Bias (Intercept)
        # b_k = -0.5 * (mu_k . W_k) + log(pi_k)
        # We compute the diagonal of (Means @ W.T) efficiently using element-wise mult and sum
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
    print("\n[3/5] Training CholeskyOASDiscriminant...")
    model = CholeskyOASDiscriminant()
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
