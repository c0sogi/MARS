import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from scipy.special import softmax
from sklearn.metrics import log_loss
import warnings

from library import config, preprocessing

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")


class DualPrecisionOAS:
    """
    Dual-Precision Gaussian-OAS Discriminant.

    Implements a custom Linear Discriminant Analysis using the OAS estimator for covariance.
    Key features:
    1. Parameter Estimation in Float64: To capture subtle covariance structures.
    2. Inference in Float32: To act as a quantization regularizer and optimize log-loss saturation.
    3. Shared Covariance: Assumes a common covariance matrix across all classes (LDA-like).
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None  # Will be stored as float32
        self.precision_ = None  # Will be stored as float32
        self.log_priors_ = None  # Will be stored as float32

    def fit(self, X, y):
        """
        Fit the model using Float64 precision, then quantize parameters to Float32.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        # Enforce Float64 for training
        X_64 = np.array(X, dtype=np.float64)

        # Identify classes and priors
        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_features = X_64.shape[1]

        # Calculate Priors (Frequency-based)
        priors = counts / float(len(y))
        log_priors_64 = np.log(priors)

        # Calculate Class Means (Centroids) in Float64
        means_64 = np.zeros((n_classes, n_features), dtype=np.float64)
        for i, cls in enumerate(self.classes_):
            means_64[i] = X_64[y == cls].mean(axis=0)

        # Calculate Centered Residuals for Covariance Estimation
        # Map each sample to its class index to subtract the correct mean
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        y_indices = np.array([class_to_idx[cls] for cls in y])
        residuals = X_64 - means_64[y_indices]

        # Estimate Covariance using OAS (Oracle Approximating Shrinkage)
        # OAS is preferred for Small-N, High-D scenarios
        oas = OAS()
        oas.fit(residuals)
        covariance_64 = oas.covariance_

        # Compute Precision Matrix (Inverse Covariance) in Float64
        # Use pseudo-inverse or standard inverse. Since OAS regularizes, standard inv is usually safe.
        # We use np.linalg.inv for strict adherence to the mathematical formulation.
        precision_64 = np.linalg.inv(covariance_64)

        # Quantization Step: Cast parameters to Float32
        self.means_ = means_64.astype(np.float32)
        self.precision_ = precision_64.astype(np.float32)
        self.log_priors_ = log_priors_64.astype(np.float32)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using Float32 precision.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities.
        """
        # Enforce Float32 for inference
        X_32 = np.array(X, dtype=np.float32)
        n_samples = X_32.shape[0]
        n_classes = len(self.classes_)

        scores = np.zeros((n_samples, n_classes), dtype=np.float32)

        # Compute Discriminant Scores
        # Score_k = -0.5 * (x - mu_k)^T P (x - mu_k) + log(pi_k)
        # Note: The term -0.5 * log|Sigma| is constant across classes for shared covariance, so it cancels out.

        for k in range(n_classes):
            # Difference vector: (N, D)
            diff = X_32 - self.means_[k]

            # Mahalanobis distance calculation
            # (N, D) @ (D, D) -> (N, D)
            temp = np.dot(diff, self.precision_)

            # Row-wise dot product: sum(diff * temp, axis=1) -> (N,)
            # This computes (x-mu)^T P (x-mu) for each sample
            dist = np.sum(diff * temp, axis=1)

            scores[:, k] = -0.5 * dist + self.log_priors_[k]

        # Apply Softmax to get probabilities
        probs = softmax(scores, axis=1)

        return probs


def train_and_predict():
    """
    Main execution pipeline:
    1. Load preprocessed data.
    2. Validate model on the validation set.
    3. Retrain on the full dataset (Train + Val).
    4. Generate predictions for the test set.
    5. Save submission.
    """
    # 1. Load Data
    # The preprocessing pipeline handles float64 conversion and feature extraction
    X_train, y_train, X_val, y_val, X_test, test_ids = (
        preprocessing.get_preprocessed_data(load_cached_data=True)
    )

    print(f"Data Loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 2. Validation Phase
    print("Training validation model...")
    model_val = DualPrecisionOAS()
    model_val.fit(X_train, y_train)

    print("Evaluating on validation set...")
    val_probs = model_val.predict_proba(X_val)

    # Calculate Log Loss
    # We use the model's classes_ to ensure correct column mapping
    val_loss = log_loss(y_val, val_probs, labels=model_val.classes_)
    print(f"Validation Multi-class Log Loss: {val_loss}")

    # 3. Full Training Phase
    # Concatenate Train and Val to maximize N for the OAS estimator
    print("Retraining on full dataset (Train + Val)...")
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    model_full = DualPrecisionOAS()
    model_full.fit(X_full, y_full)

    # 4. Submission Generation
    print("Generating test predictions...")
    test_probs = model_full.predict_proba(X_test)

    # 5. Format and Save Submission
    # Create DataFrame with IDs and Probabilities
    submission_df = pd.DataFrame(test_probs, columns=model_full.classes_)
    submission_df.insert(0, config.ID_COL, test_ids)

    # Ensure ID is integer
    submission_df[config.ID_COL] = submission_df[config.ID_COL].astype(int)

    # Save
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    train_and_predict()
