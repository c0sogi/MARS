import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from library.config import SUBMISSION_DIR, ID_COL
from library.data_loader import load_data


class ParsimoniousOASClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using the OAS Covariance Backbone.
    Designed for 'Small N, Large P' regimes by using analytical shrinkage
    and solving the linear discriminant algebraically in float64 precision.
    """

    def __init__(self, assume_centered=True):
        self.assume_centered = assume_centered
        self.classes_ = None
        self.le_ = None
        self.W_ = None  # Weights (n_classes, n_features)
        self.b_ = None  # Bias (n_classes,)
        self.precision_ = None
        self.feature_names_in_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation on centered residuals.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Target labels.
        """
        # 1. Enforce float64 precision
        X_data = np.array(X, dtype=np.float64)

        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()

        # 2. Encode Labels
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_samples, n_features = X_data.shape

        # 3. Compute Class Means and Priors
        # We use a simple loop to ensure explicit control over precision
        means = np.zeros((n_classes, n_features), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)

        for k in range(n_classes):
            mask = y_enc == k
            X_k = X_data[mask]
            means[k] = np.mean(X_k, axis=0)
            priors[k] = len(X_k) / n_samples

        # 4. Compute Residuals (Centering)
        # R = X - mu_y
        residuals = X_data - means[y_enc]

        # 5. Estimate Covariance using OAS
        # We use assume_centered=True because we manually centered the residuals
        # OAS is an analytical shrinkage estimator ideal for high-dimensional data
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        # 6. Extract Precision Matrix (Inverse Covariance)
        # Sklearn's OAS uses SVD-based pseudo-inverse for numerical stability
        self.precision_ = oas.precision_

        # 7. Derive Linear Weights and Bias
        # W = means @ precision (Shape: n_classes x n_features)
        # Note: precision is symmetric
        self.W_ = np.dot(means, self.precision_)

        # b_k = -0.5 * (mu_k^T @ P @ mu_k) + log(pi_k)
        # The term (mu_k^T @ P @ mu_k) is the dot product of mu_k and W_k
        # We compute this row-wise for all classes
        quad_term = 0.5 * np.sum(self.W_ * means, axis=1)
        self.b_ = -quad_term + np.log(priors)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linear formulation.

        Args:
            X (pd.DataFrame or np.ndarray): Features.

        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes).
        """
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model must be fitted before prediction.")

        X_data = np.array(X, dtype=np.float64)

        # Linear Projection: Z = X @ W.T + b
        # This cancels out the quadratic term x^T P x which is common to all classes
        logits = np.dot(X_data, self.W_.T) + self.b_

        # Softmax in float64
        probs = softmax(logits, axis=1)

        return probs

    def predict(self, X):
        """
        Predicts class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def run_training_pipeline(load_cached_data=True):
    """
    Orchestrates the data loading, model training, evaluation, and submission generation.
    """
    print("Starting Parsimonious OAS Pipeline...")

    # 1. Load Data
    # The data loader handles feature engineering (geometric extraction),
    # subtractive fusion, and inductive preprocessing.
    X_train, y_train, X_val, y_val, X_test, test_ids = load_data(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    model = ParsimoniousOASClassifier(assume_centered=True)

    # 3. Train Model
    print("Fitting model...")
    model.fit(X_train, y_train)

    # 4. Evaluate on Validation Set
    print("Evaluating on Validation set...")
    val_probs = model.predict_proba(X_val)

    # Clip probabilities for metric calculation stability (as per task description)
    # max(min(p, 1-1e-15), 1e-15)
    val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)

    # Calculate Multi-class Log Loss
    # We need to ensure y_val matches the order of columns in val_probs
    # val_probs columns correspond to model.classes_ (alphabetical)
    loss = log_loss(y_val, val_probs_clipped, labels=model.classes_)

    print("Validation Multi-class Log Loss:")
    print(loss)  # Print full precision

    # 5. Generate Submission
    print("Generating predictions for Test set...")
    test_probs = model.predict_proba(X_test)
    test_probs_clipped = np.clip(test_probs, 1e-15, 1 - 1e-15)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(test_probs_clipped, columns=model.classes_)
    submission_df.insert(0, ID_COL, test_ids.values)

    # Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
