import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder

from library.config import OAS_PARAMS, USE_FLOAT64, SUBMISSION_FILE, ID_COL
from library.utils import compute_log_loss, set_seed
from library.data_loader import load_and_process_data
from library.preprocessing import preprocess_data


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using OAS Covariance Estimator.

    Implements the analytic solution for LDA:
    Discriminant_k(x) = x^T * (Sigma^-1 * mu_k) - 0.5 * (mu_k^T * Sigma^-1 * mu_k) + log(pi_k)

    Attributes:
        classes_ (np.ndarray): Unique class labels.
        coef_ (np.ndarray): Weight matrix (n_classes, n_features).
        intercept_ (np.ndarray): Bias vector (n_classes,).
    """

    def __init__(self, oas_params=None):
        self.oas_params = oas_params if oas_params is not None else OAS_PARAMS
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.le_ = None

    def fit(self, X, y):
        """
        Fits the model.

        1. Computes class means and priors.
        2. Computes residuals (X - class_mean).
        3. Fits OAS on residuals to get Precision Matrix (Sigma^-1).
        4. Computes Linear Weights (W) and Bias (b).
        """
        if USE_FLOAT64:
            X = X.astype(np.float64)

        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Class Means and Priors
        means = np.zeros((n_classes, n_features), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)

        # We also need residuals for covariance estimation
        residuals = np.zeros_like(X, dtype=np.float64)

        for k in range(n_classes):
            mask = y_encoded == k
            X_k = X[mask]

            # Empirical Mean
            means[k] = np.mean(X_k, axis=0)

            # Empirical Prior
            priors[k] = len(X_k) / len(X)

            # Residuals
            residuals[mask] = X_k - means[k]

        # 2. Estimate Covariance/Precision using OAS on Residuals
        # assume_centered=True because residuals have zero mean by definition
        oas = OAS(**self.oas_params)
        oas.fit(residuals)

        # Precision Matrix (Sigma^-1)
        # SVD-based pseudo-inverse from sklearn implementation is robust
        precision = oas.precision_

        # 3. Derive Linear Boundaries
        # W_k = Sigma^-1 * mu_k
        # Shape: (n_features, n_features) @ (n_classes, n_features).T -> (n_features, n_classes)
        # Transpose to get (n_classes, n_features)
        self.coef_ = (precision @ means.T).T

        # b_k = -0.5 * (mu_k^T * Sigma^-1 * mu_k) + log(pi_k)
        # Note: (mu_k^T * W_k^T) is the dot product of mean and weight
        # We can do element-wise multiply and sum
        term1 = -0.5 * np.sum(means * self.coef_, axis=1)
        term2 = np.log(priors)
        self.intercept_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Z = X @ W.T + b
        Prob = Softmax(Z)
        """
        if USE_FLOAT64:
            X = X.astype(np.float64)

        # Linear Score
        logits = X @ self.coef_.T + self.intercept_

        # Softmax
        probs = softmax(logits, axis=1)

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def run_training_and_submission(load_cached_data=True):
    """
    Executes the full pipeline:
    1. Load Data
    2. Preprocess
    3. Train & Validate
    4. Retrain on Full Data
    5. Generate Submission
    """
    set_seed()

    # 1. Load Data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_process_data(
        load_cached_data=load_cached_data
    )

    # 2. Preprocess
    print("Preprocessing data...")
    X_train_trans, X_val_trans, X_test_trans = preprocess_data(
        X_train, X_val, X_test, load_cached_data=load_cached_data
    )

    # 3. Validation Run
    print("Training OAS Discriminant on Training Set...")
    model = OASDiscriminant()
    model.fit(X_train_trans, y_train)

    print("Validating...")
    val_probs = model.predict_proba(X_val_trans)
    val_loss = compute_log_loss(y_val, val_probs, model.classes_)
    print(f"Validation Multi-class Log Loss: {val_loss:.15f}")

    # 4. Full Retraining for Submission
    print("Retraining on combined (Train + Val) set for submission...")
    X_full = np.concatenate([X_train_trans, X_val_trans], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    model_full = OASDiscriminant()
    model_full.fit(X_full, y_full)

    # 5. Generate Submission
    print("Generating predictions for Test Set...")
    test_probs = model_full.predict_proba(X_test_trans)

    # Create Submission DataFrame
    # Columns must be the class names
    submission_df = pd.DataFrame(test_probs, columns=model_full.classes_)

    # Insert ID column at the beginning
    submission_df.insert(0, ID_COL, test_ids)

    # Save
    print(f"Saving submission to {SUBMISSION_FILE}...")
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    # This block is for local testing if run directly,
    # but the task requires implementing the module functions.
    # The pipeline execution is encapsulated in run_training_and_submission.
    pass
