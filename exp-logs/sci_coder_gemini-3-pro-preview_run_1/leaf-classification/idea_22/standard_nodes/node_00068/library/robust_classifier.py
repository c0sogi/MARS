import numpy as np
import pandas as pd
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelBinarizer

from library.config import OUTPUT_DIR, WORKING_DIR
from library.math_utils import compute_geometric_median
from library.data_manager import load_dataset


class GeometricOASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    A Robust Linear Discriminant Classifier that uses the Geometric Median for
    class centroids and the OAS estimator for the shared covariance matrix.

    This approach is robust to outliers in small-sample regimes and maintains
    high numerical precision (float64).
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Args:
            X (np.ndarray): Training vector, shape (n_samples, n_features).
            y (np.ndarray): Target values, shape (n_samples,).
        """
        # Ensure input is float64 for precision
        X, y = check_X_y(X, y, dtype=np.float64)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Robust Centroids (Geometric Median) and Residuals
        self.centroids_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        all_residuals = []

        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]

            # Compute Geometric Median
            centroid = compute_geometric_median(X_cls)
            self.centroids_[idx] = centroid

            # Compute Prior
            self.priors_[idx] = X_cls.shape[0] / X.shape[0]

            # Compute Residuals (centered on geometric median)
            residuals = X_cls - centroid
            all_residuals.append(residuals)

        # Stack all residuals to estimate global covariance
        R = np.vstack(all_residuals)

        # 2. Estimate Covariance using OAS
        # assume_centered=True because we manually centered using geometric medians
        oas = OAS(assume_centered=True)
        oas.fit(R)

        self.covariance_ = oas.covariance_
        self.precision_ = oas.precision_  # Inverse covariance

        # 3. Pre-compute Linear Decision Boundaries
        # Discriminant function: delta_k(x) = x.T * P * mu_k - 0.5 * mu_k.T * P * mu_k + log(pi_k)
        # Linear term W_k = P * mu_k
        # Bias term b_k = -0.5 * mu_k.T * W_k + log(pi_k)

        # W shape: (n_classes, n_features)
        self.W_ = np.dot(self.centroids_, self.precision_)

        # b shape: (n_classes,)
        # Compute quadratic term: diag(mu @ P @ mu.T) -> diag(mu @ W.T)
        quadratic = 0.5 * np.sum(self.centroids_ * self.W_, axis=1)
        self.b_ = -quadratic + np.log(self.priors_)

        return self

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        Z = X @ W.T + b
        """
        check_is_fitted(self, ["W_", "b_", "classes_"])
        X = check_array(X, dtype=np.float64)

        return np.dot(X, self.W_.T) + self.b_

    def predict_proba(self, X):
        """
        Probability estimation using Softmax on the decision function logits.
        """
        decision = self.decision_function(X)

        # Numerical stability for softmax: subtract max per row
        decision = decision - np.max(decision, axis=1, keepdims=True)
        exp_decision = np.exp(decision)
        proba = exp_decision / np.sum(exp_decision, axis=1, keepdims=True)

        return proba

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def run_training_pipeline(load_cached=True):
    """
    Orchestrates the data loading, model training, validation, and submission generation.
    """
    print("Initializing Robust Geometric-OAS Pipeline...")

    # 1. Load Data
    # The data_manager handles inductive preprocessing (Yeo-Johnson + StandardScaler)
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=load_cached
    )

    print(f"Data Loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"Number of classes: {len(classes)}")

    # 2. Initialize Model
    model = GeometricOASDiscriminant()

    # 3. Train on Training Set
    print("Fitting GeometricOASDiscriminant on Training Set...")
    model.fit(X_train, y_train)

    # 4. Evaluate on Validation Set
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # Note: scikit-learn's log_loss handles the clipping internally,
    # but we pass raw probabilities.
    val_loss = log_loss(y_val, val_probs, labels=model.classes_)

    print("-" * 30)
    print(f"Validation Multi-class Log Loss: {val_loss}")
    print("-" * 30)

    # 5. Generate Test Predictions
    print("Generating predictions for Test Set...")
    test_probs = model.predict_proba(X_test)

    # 6. Create Submission File
    # The submission format requires: id, class_1, class_2, ...
    # columns should be the class names.

    # Create DataFrame
    submission_df = pd.DataFrame(test_probs, columns=classes)
    submission_df.insert(0, "id", test_ids)

    # Save
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")

    # Optional: Validate submission format briefly
    print("Submission Head:")
    print(submission_df.head(2))


if __name__ == "__main__":
    # This block is not required by the prompt but useful for local testing if run directly
    run_training_pipeline()
