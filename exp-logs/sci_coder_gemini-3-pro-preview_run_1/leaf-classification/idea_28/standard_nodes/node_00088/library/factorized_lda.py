import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from library import config, preprocessing


class GlobalOASLDA:
    """
    Global High-Precision OAS Discriminant.

    Uses the full feature space to estimate a global covariance matrix via OAS.
    Cite Lesson 87: Prefer global shrinkage estimators over manual feature factorization.
    Cite Lesson 47: Prefer OAS over Ledoit-Wolf on Gaussianized Data.
    """

    def __init__(self):
        self.W = None
        self.b = None
        self.le = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits the global LDA model.

        Args:
            X (array-like): Feature matrix (n_samples, n_features).
            y (array-like): Target labels.
        """
        # Encode labels
        y_enc = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Compute Priors
        counts = np.bincount(y_enc)
        priors = counts / n_samples
        # Cite Lesson 33: Use empirical priors for stratified data
        priors = np.maximum(priors, 1e-15)
        log_priors = np.log(priors).astype(config.FLOAT_PRECISION)

        print(
            f"Training Global OAS LDA on {n_samples} samples with {n_features} features..."
        )

        # 1. Compute Class Means
        # Shape: (n_classes, n_features)
        means = np.zeros((n_classes, n_features), dtype=config.FLOAT_PRECISION)
        for c in range(n_classes):
            means[c] = X[y_enc == c].mean(axis=0)

        # 2. Compute Centered Residuals
        # X_centered = X - mu_y
        X_centered = X - means[y_enc]

        # 3. Estimate Precision Matrix via OAS
        # Cite Lesson 61: assume_centered=True ensures geometric consistency
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)

        # Cite Lesson 62: Use precision_ attribute (SVD-based pseudo-inverse)
        precision = oas.precision_.astype(config.FLOAT_PRECISION)

        # 4. Compute Linear Weights and Biases (Linear Formulation)
        # Cite Lesson 55: Prefer Linear Formulation over distance-based
        # W = Precision @ Means.T  -> Shape (n_features, n_classes)
        self.W = precision @ means.T

        # Bias term: b_k = -0.5 * (mu_k.T @ Sigma^-1 @ mu_k) + log(pi_k)
        # (means @ precision) is (n_classes, n_features)
        # Multiply element-wise with means -> sum over features
        quad_term = -0.5 * np.sum((means @ precision) * means, axis=1)

        self.b = quad_term + log_priors

        print(f"Fitted Global OAS (Precision shape: {precision.shape})")
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Probability matrix.
        """
        # Linear discriminant: Z = XW + b
        # Cite Lesson 57: Inference precision dictates performance ceiling
        Z = X @ self.W + self.b

        # Apply Softmax
        # Cite Lesson 85: Use optimized library primitives (scipy.special.softmax)
        probs = softmax(Z, axis=1)
        return probs


def run_training_and_submission(debug_size=config.DEBUG_SAMPLE_SIZE):
    """
    Orchestrates the training, validation, and submission process.

    Args:
        debug_size (int or None): Limit dataset size for debugging.
    """
    # 1. Load Data
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, ids_test = (
        preprocessing.get_preprocessed_data(
            load_cached_data=True, debug_size=debug_size
        )
    )

    # 2. Train Model
    model = FactorizedOASLDA()
    model.fit(X_train, y_train)

    # 3. Validate
    print("Validating...")
    probs_val = model.predict_proba(X_val)

    # Calculate Log Loss
    # Clip probabilities to avoid log(0) and match competition metric logic
    eps = 1e-15
    probs_val_clipped = np.clip(probs_val, eps, 1 - eps)

    # Get true class indices
    y_val_indices = model.le.transform(y_val)

    # Select probabilities corresponding to true classes
    # Using advanced indexing: probs[row, col]
    true_class_probs = probs_val_clipped[np.arange(len(y_val)), y_val_indices]

    log_loss = -np.mean(np.log(true_class_probs))
    print(f"Validation Log Loss: {log_loss}")

    # 4. Generate Submission
    print("Generating submission...")
    probs_test = model.predict_proba(X_test)

    # Create DataFrame
    submission = pd.DataFrame(probs_test, columns=model.classes_)
    submission.insert(0, config.ID_COLUMN, ids_test)

    # Save
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
