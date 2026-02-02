import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.covariance import OAS
from sklearn.metrics import accuracy_score

from library.config import (
    FLOAT_PRECISION,
    MODEL_PARAMS,
    SEED,
    SUBMISSION_DIR,
    SAMPLE_SUBMISSION_FILE,
)
from library.utils import set_seed, calculate_log_loss
from library.preprocessing import SanitizedGroupPreprocessor


class FactorizedOASLDA:
    """
    A Factorized Linear Discriminant Analysis model using OAS covariance estimation.

    This model trains independent LDA experts on different feature groups and
    aggregates them via Bayesian logit summation. It uses Oracle Approximating
    Shrinkage (OAS) to estimate covariance matrices, which is robust for
    high-dimensional, small-sample datasets.
    """

    def __init__(self):
        self.groups = None
        self.group_models = {}  # Stores (Weights, Biases) for each group
        self.priors = None
        self.classes_ = None
        self.dtype = FLOAT_PRECISION

    def _fit_group(self, X, y):
        """
        Fits an OAS-LDA expert on a single feature group.

        Args:
            X (np.ndarray): Feature matrix for the group (n_samples, n_features).
            y (np.ndarray): Target labels (n_samples,).

        Returns:
            tuple: (W, b) where W is the weight matrix and b is the bias vector.
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Initialize containers
        means = np.zeros((n_classes, n_features), dtype=self.dtype)
        residuals = np.zeros_like(X, dtype=self.dtype)

        # 1. Compute Class Means and Residuals
        # Residuals = X - class_mean, used for pooled covariance estimation
        for k in range(n_classes):
            mask = y == k
            X_k = X[mask]
            if len(X_k) > 0:
                mu_k = np.mean(X_k, axis=0, dtype=self.dtype)
                means[k] = mu_k
                residuals[mask] = X_k - mu_k
            else:
                # Fallback for empty classes (unlikely in stratified splits)
                pass

        # 2. Estimate Covariance/Precision using OAS
        # assume_centered=True because we manually centered using class means
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        # Extract precision matrix (inverse covariance)
        precision = oas.precision_.astype(self.dtype)

        # 3. Compute Linear Discriminant Parameters
        # LDA Discriminant: d_k(x) = x.T @ (P @ mu_k) - 0.5 * (mu_k.T @ P @ mu_k) + log(pi_k)
        # We calculate W = P @ mu.T so that x @ W gives the first term.

        # W shape: (n_features, n_classes)
        W = np.dot(precision, means.T)

        # Quadratic term: -0.5 * diag(means @ W)
        # We compute only the diagonal efficiently
        quad_term = -0.5 * np.sum(means * W.T, axis=1)  # Shape (n_classes,)

        # Add log priors to the bias
        # Note: This means each expert includes the prior. We will correct this during aggregation.
        log_priors = np.log(self.priors)
        b = quad_term + log_priors

        return W, b

    def fit(self, X_dict, y):
        """
        Fits the ensemble on the dictionary of feature groups.

        Args:
            X_dict (dict): Dictionary mapping group names to feature arrays.
            y (np.ndarray): Target labels.
        """
        self.groups = list(X_dict.keys())
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Compute Global Priors
        class_counts = np.bincount(y, minlength=n_classes)
        # Add epsilon to avoid log(0) if a class is missing (though unlikely)
        self.priors = (class_counts.astype(self.dtype) + 1e-15) / class_counts.sum()

        print(f"Training FactorizedOASLDA on {len(self.groups)} groups: {self.groups}")

        for group, X in X_dict.items():
            # Ensure double precision
            X_group = X.astype(self.dtype)
            W, b = self._fit_group(X_group, y)
            self.group_models[group] = (W, b)

        return self

    def predict_proba(self, X_dict):
        """
        Predicts class probabilities by aggregating expert logits.

        Args:
            X_dict (dict): Dictionary mapping group names to feature arrays.

        Returns:
            np.ndarray: Probability matrix (n_samples, n_classes).
        """
        # Determine shapes
        first_group = self.groups[0]
        n_samples = X_dict[first_group].shape[0]
        n_classes = len(self.classes_)

        # Initialize total logits
        total_logits = np.zeros((n_samples, n_classes), dtype=self.dtype)

        # Sum logits from all experts
        for group in self.groups:
            X = X_dict[group].astype(self.dtype)
            W, b = self.group_models[group]

            # Linear Logit: Z = X @ W + b
            logits = np.dot(X, W) + b
            total_logits += logits

        # Bayesian Aggregation Correction
        # Since each of the G experts adds log(prior), the sum contains G * log(prior).
        # We need the final posterior to contain exactly 1 * log(prior).
        # Therefore, we subtract (G - 1) * log(prior).
        if MODEL_PARAMS.get("logit_aggregation_correction", True):
            G = len(self.groups)
            log_priors = np.log(self.priors)
            correction = (G - 1) * log_priors
            total_logits -= correction

        # Apply Softmax to get probabilities
        return softmax(total_logits, axis=1)


def run_model():
    """
    Main execution function.
    1. Loads and preprocesses data (with caching).
    2. Trains the FactorizedOASLDA model.
    3. Evaluates on the validation set.
    4. Generates predictions for the test set.
    5. Saves the submission file.
    """
    set_seed(SEED)

    print("Initializing Data Preprocessing...")
    preprocessor = SanitizedGroupPreprocessor()

    # Load data (calculates features and transforms if not cached)
    data = preprocessor.process_and_cache(load_cached_data=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    test_ids = data["test_ids"]
    classes = data["classes"]

    print(f"\nData Loaded. Train samples: {len(y_train)}, Val samples: {len(y_val)}")

    # Initialize and Train Model
    model = FactorizedOASLDA()
    model.fit(X_train, y_train)

    # Validation
    print("\nRunning Validation...")
    y_pred_val = model.predict_proba(X_val)

    # Calculate Metrics
    # Note: calculate_log_loss handles clipping and normalization internally
    val_loss = calculate_log_loss(y_val, y_pred_val, labels=list(range(len(classes))))

    # Calculate Accuracy for reference
    y_pred_labels = np.argmax(y_pred_val, axis=1)
    val_acc = accuracy_score(y_val, y_pred_labels)

    print(f"Validation Log Loss: {val_loss:.15f}")
    print(f"Validation Accuracy: {val_acc:.6f}")

    # Test Prediction
    print("\nGenerating Test Predictions...")
    y_pred_test = model.predict_proba(X_test)

    # Prepare Submission
    # Create DataFrame
    submission_df = pd.DataFrame(y_pred_test, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Verify Submission Format against Sample
    if os.path.exists(SAMPLE_SUBMISSION_FILE):
        sample_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)
        print(f"Sample submission shape: {sample_df.shape}")
        print(f"Generated submission shape: {submission_df.shape}")
        # Basic check on columns
        missing_cols = set(sample_df.columns) - set(submission_df.columns)
        if not missing_cols:
            print("Column verification passed.")
        else:
            print(f"Warning: Missing columns in submission: {missing_cols}")
