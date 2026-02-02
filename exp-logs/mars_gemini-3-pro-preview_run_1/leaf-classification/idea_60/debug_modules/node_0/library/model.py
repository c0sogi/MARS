import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from scipy.special import softmax
from library import config, utils, data


class SanitizedOASDiscriminant:
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS).
    Designed for high-precision analytical inference using float64 arithmetic.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.covariance_estimator_ = None
        self.W_ = None
        self.b_ = None

    def fit(self, X, y):
        """
        Fits the OAS-LDA model.
        1. Computes class means and priors.
        2. Computes residuals (X - mu_y).
        3. Fits OAS on residuals to estimate covariance/precision.
        4. Derives linear weights and biases.
        """
        # Ensure float64 precision
        X = X.astype(np.float64)

        # 1. Parameter Estimation
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        # Calculate means and priors
        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.means_[idx] = np.mean(X_cls, axis=0)
            self.priors_[idx] = len(X_cls) / len(X)

        # 2. Compute Residuals
        # Map y to indices 0..K-1 for indexing means
        # Assuming y is already label-encoded 0..K-1 by DataManager, but being safe:
        y_indices = np.searchsorted(self.classes_, y)
        residuals = X - self.means_[y_indices]

        # 3. Estimate Covariance (OAS)
        # assume_centered=True because we explicitly centered via residuals
        self.covariance_estimator_ = OAS(assume_centered=True)
        self.covariance_estimator_.fit(residuals)

        # 4. Derive Weights and Bias
        # Precision matrix P = Sigma^-1
        P = self.covariance_estimator_.precision_

        # W = P * mu (shape: D x K, but we store as K x D for W_k * x)
        # W_k = mu_k^T P. Since P is symmetric, W = means @ P
        self.W_ = self.means_ @ P  # Shape: (n_classes, n_features)

        # Bias b_k = -0.5 * (mu_k^T P mu_k) + log(pi_k)
        # The quadratic term is the diagonal of (means @ P @ means.T)
        # Or more efficiently: sum(W * means, axis=1)
        quadratic_term = np.sum(self.W_ * self.means_, axis=1)
        self.b_ = -0.5 * quadratic_term + np.log(self.priors_)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using linear inference.
        Z = X W^T + b
        """
        X = X.astype(np.float64)

        # Linear Inference
        logits = X @ self.W_.T + self.b_

        # Softmax
        probs = softmax(logits, axis=1)
        return probs

    def predict(self, X):
        """
        Predicts class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def run_training_pipeline(load_cached_data=True, debug_limit=None):
    """
    Orchestrates the data loading, training, validation, and submission generation.
    """
    with utils.Timer("Full Pipeline Execution"):
        # 1. Data Preparation
        data_manager = data.LeafDataManager()
        X_train, y_train, X_val, y_val, X_test, test_ids, class_names = (
            data_manager.prepare_data(
                load_cached_data=load_cached_data, debug_limit=debug_limit
            )
        )

        utils.Logger.info(f"Training Data Shape: {X_train.shape}")
        utils.Logger.info(f"Validation Data Shape: {X_val.shape}")
        utils.Logger.info(f"Test Data Shape: {X_test.shape}")

        # 2. Model Training
        utils.Logger.info("Initializing Sanitized OAS Discriminant...")
        model = SanitizedOASDiscriminant()

        with utils.Timer("Model Fitting"):
            model.fit(X_train, y_train)

        # 3. Validation
        utils.Logger.info("Evaluating on Validation Set...")
        val_probs = model.predict_proba(X_val)

        # Clip probabilities for metric calculation stability (mimicking competition metric)
        val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
        # Rescale rows to sum to 1 after clipping
        val_probs_clipped /= val_probs_clipped.sum(axis=1, keepdims=True)

        val_loss = log_loss(y_val, val_probs_clipped)
        utils.Logger.metric("Validation Multi-class Log Loss", val_loss)

        # 4. Submission Generation
        utils.Logger.info("Generating predictions for Test Set...")
        test_probs = model.predict_proba(X_test)

        # Apply strict clipping for submission as per task description
        test_probs = np.maximum(np.minimum(test_probs, 1 - 1e-15), 1e-15)
        # Note: The prompt says probabilities are rescaled by the scorer,
        # but we should provide valid probabilities.

        # Create Submission DataFrame
        submission_df = pd.DataFrame(test_probs, columns=class_names)
        submission_df.insert(0, config.ID_COL, test_ids)

        # Save Submission
        utils.save_cache_parquet(
            os.path.join(config.WORKING_DIR, "submission_debug.parquet"), submission_df
        )

        # Save to final submission CSV
        if not os.path.exists(config.SUBMISSION_DIR):
            os.makedirs(config.SUBMISSION_DIR)

        submission_df.to_csv(config.SUBMISSION_FILE_PATH, index=False)
        utils.Logger.info(f"Submission saved to {config.SUBMISSION_FILE_PATH}")
