import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import os

from library.config import FLOAT_PRECISION, SEED, SUBMISSION_DIR, ID_COL, TARGET_COL
from library.data_loader import load_and_process_data
from library.preprocessing import get_preprocessed_data


class RatioProjectedOAS:
    """
    Custom Linear Discriminant Classifier using OAS for covariance estimation
    and explicit ratio-projected features.

    This model implements the 'Linearized Inference' strategy to algebraically
    cancel unstable quadratic terms found in distance-based formulations,
    relying on the SVD-based precision matrix from the OAS estimator.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.precision_ = None
        self.W_ = None
        self.b_ = None
        self.estimator = None
        self.le = None

    def fit(self, X, y):
        """
        Fits the model using High-Precision OAS on class-centered residuals.

        Args:
            X: Training features (numpy array or DataFrame).
            y: Training labels.
        """
        # Enforce strict float64 precision
        X = X.astype(FLOAT_PRECISION)

        # Encode labels
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Empirical Means and Priors
        self.means_ = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        for k in range(n_classes):
            mask = y_encoded == k
            X_k = X[mask]

            if len(X_k) > 0:
                self.means_[k] = np.mean(X_k, axis=0)
                self.priors_[k] = len(X_k) / len(X)
            else:
                # Fallback for empty classes (should not happen with stratified split)
                self.means_[k] = 0.0
                self.priors_[k] = 0.0

        # 2. Compute Centered Residuals (R = X - mu_y)
        # This aligns all classes to the origin to estimate the common shape (covariance)
        R = np.zeros_like(X, dtype=FLOAT_PRECISION)
        for k in range(n_classes):
            mask = y_encoded == k
            if np.any(mask):
                R[mask] = X[mask] - self.means_[k]

        # 3. Estimate Covariance with OAS
        # assume_centered=True because R is explicitly centered
        self.estimator = OAS(assume_centered=True)
        self.estimator.fit(R)

        # 4. Extract Precision Matrix (Inverse Covariance)
        self.precision_ = self.estimator.precision_.astype(FLOAT_PRECISION)

        # 5. Derive Linear Weights and Bias
        # The decision boundary is linear: Z = X @ W.T + b
        # W = mu @ P (Shape: K x D) - Note: P is symmetric
        self.W_ = np.dot(self.means_, self.precision_)

        # b = -0.5 * diag(mu @ P @ mu^T) + log(priors)
        # Efficiently computed as: -0.5 * sum(mu * W, axis=1) + log(priors)
        quad_term = 0.5 * np.sum(self.means_ * self.W_, axis=1)

        # Add epsilon to priors to avoid log(0)
        self.b_ = -quad_term + np.log(self.priors_ + 1e-15)

        return self

    def predict_proba(self, X):
        """
        Predicts probabilities using the linearized discriminant function.
        """
        X = X.astype(FLOAT_PRECISION)

        # Linear Score: Z = X @ W.T + b
        Z = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax with high precision
        return softmax(Z, axis=1)

    def predict(self, X):
        """
        Predicts class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def run_task():
    """
    Main execution function to load data, train model, and generate submission.
    """
    # 1. Load Data
    # The data loader handles caching of the raw geometric extraction and ratio computation
    print("Loading data...")
    (
        (X_train_raw, y_train, train_ids),
        (X_val_raw, y_val, val_ids),
        (X_test_raw, test_ids),
    ) = load_and_process_data(load_cached_data=True)

    # 2. Preprocess
    # The preprocessor handles caching of the Yeo-Johnson + Scaling transformation
    print("Preprocessing data...")
    X_train, X_val, X_test = get_preprocessed_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    # 3. Train Model
    print("Training Ratio-Projected OAS Discriminant...")
    model = RatioProjectedOAS()
    model.fit(X_train, y_train)

    # 4. Validate
    print("Validating...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # We pass labels explicitly to ensure correct column mapping
    loss = log_loss(y_val, val_probs, labels=model.classes_)
    print(f"Validation Multi-class Log Loss: {loss:.15f}")

    # 5. Predict on Test
    print("Generating Test Predictions...")
    test_probs = model.predict_proba(X_test)

    # 6. Format Submission
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)
    submission_df.insert(0, ID_COL, test_ids)

    # Ensure output directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
