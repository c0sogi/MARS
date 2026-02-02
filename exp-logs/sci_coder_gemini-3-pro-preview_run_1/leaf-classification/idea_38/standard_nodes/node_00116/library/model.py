import numpy as np
import pandas as pd
import os
import warnings
from scipy.special import softmax
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from library.config import (
    SEED,
    FLOAT_PRECISION,
    OAS_PARAMS,
    CLIP_EPSILON,
    SUBMISSION_PATH,
)
from library.preprocessing import get_preprocessed_data

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set global seed for reproducibility
np.random.seed(SEED)


class OASLinearDiscriminant:
    """
    Custom Linear Discriminant Analysis using Oracle Approximating Shrinkage (OAS)
    for covariance estimation. Implements exact linear inference in float64.
    """

    def __init__(self):
        self.classes_ = None
        self.W_ = None  # Weight matrix (n_classes, n_features)
        self.b_ = None  # Bias vector (n_classes,)
        self.precision_ = None
        self.covariance_ = None

    def fit(self, X, y):
        """
        Fits the model by estimating class means, priors, and the shared
        covariance matrix using OAS on the residuals.
        """
        # Enforce high precision
        X = np.array(X, dtype=FLOAT_PRECISION)

        # Identify and sort classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Map string labels to integer indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[yi] for yi in y])

        # Initialize statistics
        means = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        priors = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        # Compute empirical means and priors
        for k in range(n_classes):
            X_k = X[y_idx == k]
            means[k] = np.mean(X_k, axis=0)
            priors[k] = X_k.shape[0] / n_samples

        # Compute residuals (center data by class mean)
        # This removes the class-specific mean, leaving the intra-class variation
        X_residuals = X - means[y_idx]

        # Estimate Covariance Matrix using OAS
        # assume_centered=True is correct because X_residuals has zero mean
        estimator = OAS(**OAS_PARAMS)
        estimator.fit(X_residuals)

        self.covariance_ = estimator.covariance_.astype(FLOAT_PRECISION)
        self.precision_ = estimator.precision_.astype(FLOAT_PRECISION)

        # Derive Linear Decision Boundaries
        # We compute the linear discriminant functions:
        # delta_k(x) = x.T @ (P @ mu_k) - 0.5 * (mu_k.T @ P @ mu_k) + log(pi_k)
        # This can be vectorized as Z = X @ W.T + b

        # W (n_classes, n_features) where row k is (P @ mu_k).T
        self.W_ = np.dot(means, self.precision_)

        # b (n_classes,)
        self.b_ = np.zeros(n_classes, dtype=FLOAT_PRECISION)
        for k in range(n_classes):
            # Quadratic term: -0.5 * mu_k.T @ P @ mu_k
            quad_term = -0.5 * np.dot(np.dot(means[k], self.precision_), means[k])
            log_prior = np.log(priors[k])
            self.b_[k] = quad_term + log_prior

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linear decision function and softmax.
        """
        X = np.array(X, dtype=FLOAT_PRECISION)

        # Linear Projection: Z = X @ W.T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax
        probs = softmax(logits, axis=1)

        # Clip probabilities to avoid log(0) issues in evaluation
        probs = np.clip(probs, CLIP_EPSILON, 1.0 - CLIP_EPSILON)

        return probs

    def predict(self, X):
        """
        Predicts class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def train_and_predict(load_cached_data=True, limit=None):
    """
    Main execution function to train the model, evaluate on validation,
    and generate the submission file.
    """
    print("Loading and preprocessing data...")
    # Retrieve data from the preprocessing pipeline
    (train_data, val_data, test_data) = get_preprocessed_data(
        load_cached_data=load_cached_data, limit=limit
    )

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, _, ids_test = test_data

    # Initialize and Fit Model
    print(
        f"Fitting OASLinearDiscriminant on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
    )
    model = OASLinearDiscriminant()
    model.fit(X_train, y_train)

    # Evaluation on Validation Set
    print("Evaluating on Validation set...")

    # Filter validation set to only include classes present in the training model
    # This prevents errors in debug mode where train/val sets might be disjoint
    valid_mask = np.isin(y_val, model.classes_)
    if np.sum(valid_mask) < len(y_val):
        print(
            f"DEBUG: Filtering validation set from {len(y_val)} to {np.sum(valid_mask)} samples due to class mismatch."
        )
        X_val_filtered = X_val[valid_mask]
        y_val_filtered = y_val[valid_mask]
    else:
        X_val_filtered = X_val
        y_val_filtered = y_val

    if len(y_val_filtered) > 0:
        val_probs = model.predict_proba(X_val_filtered)
        # Compute Log Loss
        # labels parameter ensures correct mapping of columns to classes
        loss = log_loss(y_val_filtered, val_probs, labels=model.classes_)
        print(f"Validation Multi-class Log Loss: {loss}")
    else:
        print(
            "Warning: Validation set empty after filtering classes. Skipping evaluation."
        )
        loss = 0.0

    # Prediction on Test Set
    print("Generating predictions for Test set...")
    test_probs = model.predict_proba(X_test)

    # Create Submission DataFrame
    # Ensure columns match the sorted class names
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)

    # Insert ID column at the beginning
    submission_df.insert(0, "id", ids_test)

    # Save Submission
    print(f"Saving submission file to {SUBMISSION_PATH}...")
    # Ensure parent directory exists (handled by config, but good practice)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print("Process completed successfully.")
    return loss
