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


class GlobalOASLDA:
    """
    A Global Linear Discriminant Analysis model using OAS covariance estimation.

    This model concatenates all feature groups into a single matrix and trains
    one global LDA model. It uses Oracle Approximating Shrinkage (OAS) to
    estimate the precision matrix, leveraging cross-feature correlations
    that factorized models miss (Cite solution_lesson_node_00204).
    """

    def __init__(self):
        self.W = None
        self.b = None
        self.priors = None
        self.classes_ = None
        self.dtype = FLOAT_PRECISION
        self.group_order = None

    def _concat_features(self, X_dict):
        """
        Concatenates features from the dictionary in a deterministic order.
        """
        if self.group_order is None:
            self.group_order = sorted(X_dict.keys())

        # Collect arrays in order
        arrays = [X_dict[g].astype(self.dtype) for g in self.group_order]
        return np.hstack(arrays)

    def fit(self, X_dict, y):
        """
        Fits the global OAS-LDA model.

        Args:
            X_dict (dict): Dictionary mapping group names to feature arrays.
            y (np.ndarray): Target labels.
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # 1. Concatenate Features
        X_global = self._concat_features(X_dict)
        n_samples, n_features = X_global.shape

        print(
            f"Training GlobalOASLDA on {n_features} features (concatenated from {len(X_dict)} groups)."
        )

        # 2. Compute Priors
        class_counts = np.bincount(y, minlength=n_classes)
        self.priors = (class_counts.astype(self.dtype) + 1e-15) / class_counts.sum()

        # 3. Compute Class Means and Residuals
        means = np.zeros((n_classes, n_features), dtype=self.dtype)
        residuals = np.zeros_like(X_global, dtype=self.dtype)

        for k in range(n_classes):
            mask = y == k
            X_k = X_global[mask]
            if len(X_k) > 0:
                mu_k = np.mean(X_k, axis=0, dtype=self.dtype)
                means[k] = mu_k
                residuals[mask] = X_k - mu_k

        # 4. Estimate Precision using OAS
        # assume_centered=True because we manually calculated residuals
        oas = OAS(assume_centered=True)
        oas.fit(residuals)
        precision = oas.precision_.astype(self.dtype)

        # 5. Compute Linear Discriminant Weights
        # W = P @ mu.T
        self.W = np.dot(precision, means.T)

        # 6. Compute Bias
        # b = -0.5 * diag(mu @ W) + log(prior)
        quad_term = -0.5 * np.sum(means * self.W.T, axis=1)
        self.b = quad_term + np.log(self.priors)

        return self

    def predict_proba(self, X_dict):
        """
        Predicts class probabilities.
        """
        X_global = self._concat_features(X_dict)

        # Linear Logit: Z = X @ W + b
        logits = np.dot(X_global, self.W) + self.b

        return softmax(logits, axis=1)


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
