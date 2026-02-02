import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from scipy.special import softmax

from library import config
from library import data_loader


class ParsimoniousOASDiscriminant:
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS).

    Designed for high-precision inference on geometric and tabular features.
    Computes the linear decision boundaries algebraically from the precision matrix
    to avoid distance-based instabilities.
    """

    def __init__(self):
        self.classes_ = None
        self.W_ = None
        self.b_ = None
        self.le_ = None
        self.precision_ = None
        self.covariance_ = None
        self.means_ = None
        self.priors_ = None

    def fit(self, X, y):
        """
        Fits the OAS Discriminant model.

        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features).
            y (np.ndarray): Target labels (n_samples,).

        Returns:
            self
        """
        # Enforce Double Precision
        X = X.astype(config.FLOAT_TYPE)

        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Initialize statistics
        self.means_ = np.zeros((n_classes, n_features), dtype=config.FLOAT_TYPE)
        self.priors_ = np.zeros(n_classes, dtype=config.FLOAT_TYPE)

        # Compute Class Means and Residuals
        # We compute residuals R = X - mu_y to estimate the common covariance matrix
        residuals = np.zeros_like(X, dtype=config.FLOAT_TYPE)

        for k in range(n_classes):
            mask = y_encoded == k
            X_k = X[mask]

            # Prior
            N_k = X_k.shape[0]
            self.priors_[k] = N_k / n_samples

            # Mean
            mu_k = np.mean(X_k, axis=0)
            self.means_[k] = mu_k

            # Centered Residuals
            residuals[mask] = X_k - mu_k

        # Estimate Covariance using OAS
        # assume_centered=True is used because we have explicitly centered the data via residuals
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        self.covariance_ = oas.covariance_
        self.precision_ = oas.precision_

        # Derive Linear Decision Boundaries
        # Log-posterior: delta_k(x) = x.T * (Sigma^-1 * mu_k) - 0.5 * (mu_k.T * Sigma^-1 * mu_k) + log(pi_k)
        # We formulate this as: Z = X * W.T + b

        # Weights: W = Means @ Precision (Result shape: n_classes x n_features)
        # Note: Precision is symmetric, so P * mu_k is equivalent to mu_k @ P
        self.W_ = np.dot(self.means_, self.precision_)

        # Bias term calculation
        # Quadratic term: -0.5 * diag(mu_k @ Sigma^-1 @ mu_k.T)
        # This is equivalent to -0.5 * sum(means * W, axis=1)
        quadratic_term = -0.5 * np.sum(self.means_ * self.W_, axis=1)
        log_prior_term = np.log(self.priors_)

        self.b_ = quadratic_term + log_prior_term

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linearized discriminant function.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probability matrix (n_samples, n_classes).
        """
        X = X.astype(config.FLOAT_TYPE)

        # Compute Logits: Z = X @ W.T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax to get probabilities
        probs = softmax(logits, axis=1)

        return probs


def run_training_pipeline(load_cached_data=True):
    """
    Executes the full training, validation, and submission pipeline.

    1. Loads data using library.data_loader.
    2. Preprocesses data (Yeo-Johnson + Scaling) using library.data_loader.
    3. Trains the ParsimoniousOASDiscriminant model.
    4. Evaluates on the validation set.
    5. Generates predictions for the test set and saves to CSV.

    Args:
        load_cached_data (bool): Whether to use cached intermediate data.

    Returns:
        tuple: (model, validation_log_loss)
    """
    print("Loading dataset...")
    # 1. Load Data
    (train_data, val_data, test_data) = data_loader.load_dataset(
        load_cached_data=load_cached_data
    )
    X_train_raw, y_train, ids_train = train_data
    X_val_raw, y_val, ids_val = val_data
    X_test_raw, ids_test = test_data

    print("Preprocessing data (High-Precision Pipeline)...")
    # 2. Preprocess Data
    X_train, X_val, X_test = data_loader.preprocess_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=load_cached_data
    )

    print("Training ParsimoniousOASDiscriminant...")
    # 3. Train Model
    model = ParsimoniousOASDiscriminant()
    model.fit(X_train, y_train)

    # 4. Validate
    print("Validating model...")
    y_pred_probs_val = model.predict_proba(X_val)

    # Calculate Multi-class Log Loss
    # We pass model.classes_ to ensure correct column alignment
    val_loss = log_loss(y_val, y_pred_probs_val, labels=model.classes_)
    print(f"Validation Multi-class Log Loss: {val_loss}")

    # 5. Generate Submission
    print("Generating submission...")
    y_pred_probs_test = model.predict_proba(X_test)

    # Construct Submission DataFrame
    submission_df = pd.DataFrame(y_pred_probs_test, columns=model.classes_)
    submission_df.insert(0, config.ID_COL, ids_test)

    # Save to disk
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return model, val_loss
