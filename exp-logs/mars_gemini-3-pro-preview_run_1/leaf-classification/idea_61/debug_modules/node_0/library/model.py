import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax

from library.config import FLOAT_PRECISION, SUBMISSION_FILE, ID_COL
from library.utils import set_seed, compute_log_loss
from library.data import load_and_merge_data
from library.pipeline import run_pipeline


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier with OAS Covariance Backbone.
    Designed for high-precision analytic inference using float64 arithmetic.
    """

    def __init__(self):
        self.classes_ = None
        self.le_ = None
        self.W_ = None
        self.b_ = None
        self.precision_matrix_ = None
        self.priors_ = None
        self.means_ = None

    def fit(self, X, y):
        """
        Fits the OAS Discriminant model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        # Enforce high precision
        X = np.array(X, dtype=FLOAT_PRECISION)
        y = np.array(y)

        # 1. Encode Labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 2. Compute Empirical Class Means and Priors
        self.means_ = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        # We need residuals for covariance estimation (X - mu_y)
        residuals = np.zeros_like(X, dtype=FLOAT_PRECISION)

        for k in range(n_classes):
            mask = y_encoded == k
            X_k = X[mask]

            # Arithmetic Mean
            mean_k = np.mean(X_k, axis=0)
            self.means_[k] = mean_k

            # Prior
            self.priors_[k] = len(X_k) / len(X)

            # Center data for this class to get residuals
            residuals[mask] = X_k - mean_k

        # 3. Estimate Covariance using OAS
        # We assume centered data because we manually computed residuals
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        # 4. Extract Precision Matrix (Inverse Covariance)
        # Use the SVD-based pseudo-inverse provided by sklearn for stability
        self.precision_matrix_ = oas.precision_.astype(FLOAT_PRECISION)

        # 5. Derive Linear Decision Boundaries (Weights and Bias)
        # The linear discriminant function is: delta_k(x) = x^T P mu_k - 0.5 mu_k^T P mu_k + log(pi_k)
        # We rewrite as: x^T W_k^T + b_k

        # W = P * mu (Shape: n_classes x n_features)
        # Note: In standard notation W is usually (n_features, n_classes), but we store as (n_classes, n_features)
        # to align with sklearn's linear_model convention (coef_).
        # Calculation: means is (n_classes, n_features), precision is (n_features, n_features)
        self.W_ = np.dot(self.means_, self.precision_matrix_)

        # b = -0.5 * diag(mu * W^T) + log(prior)
        # Compute quadratic term: 0.5 * (mu_k^T * P * mu_k)
        # Efficiently: sum(means * W, axis=1) -> dot product per class
        quad_term = 0.5 * np.sum(self.means_ * self.W_, axis=1)

        self.b_ = -quad_term + np.log(self.priors_)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linearized discriminant function.
        """
        X = np.array(X, dtype=FLOAT_PRECISION)

        # Linear Logits: Z = X W^T + b
        # X: (n_samples, n_features)
        # W_: (n_classes, n_features) -> W_.T: (n_features, n_classes)
        # b_: (n_classes,)
        # Result Z: (n_samples, n_classes)
        logits = np.dot(X, self.W_.T) + self.b_

        # Apply Softmax strictly in float64
        probs = softmax(logits, axis=1)

        return probs

    def predict(self, X):
        """
        Predicts class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def run_training_process(load_cached_data=True, debug=False):
    """
    Orchestrates the data loading, pipeline execution, training, evaluation,
    and submission generation.

    Args:
        load_cached_data (bool): Whether to use cached intermediate files.
        debug (bool): If True, runs on a small subset of data.
    """
    set_seed(42)

    print("=== Starting Sanitized Orthogonal-Geometric OAS Discriminant Pipeline ===")

    # 1. Load Data
    print("\n[1/5] Loading and Merging Data...")
    X_train_raw, y_train, train_ids = load_and_merge_data(
        "train", load_cached_data=load_cached_data
    )
    X_val_raw, y_val, val_ids = load_and_merge_data(
        "val", load_cached_data=load_cached_data
    )
    X_test_raw, _, test_ids = load_and_merge_data(
        "test", load_cached_data=load_cached_data
    )

    if debug:
        print("DEBUG MODE: Subsampling data...")
        X_train_raw = X_train_raw.iloc[:100]
        y_train = y_train[:100]
        X_val_raw = X_val_raw.iloc[:50]
        y_val = y_val[:50]

    # 2. Pipeline Processing
    print("\n[2/5] Running Sanitized High-Precision Pipeline...")
    X_train, X_val, X_test = run_pipeline(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=load_cached_data
    )

    print(
        f"Data Shapes after Pipeline: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )

    # 3. Model Training
    print("\n[3/5] Fitting OAS Discriminant...")
    model = OASDiscriminant()
    model.fit(X_train, y_train)
    print("Model fitted successfully.")

    # 4. Evaluation
    print("\n[4/5] Evaluating on Validation Set...")
    # Predict probabilities
    val_probs = model.predict_proba(X_val)

    # Convert string labels to indices for log loss calculation
    y_val_indices = model.le_.transform(y_val)

    val_loss = compute_log_loss(y_val_indices, val_probs)

    # Also compute accuracy for sanity check
    val_preds = model.predict(X_val)
    accuracy = np.mean(val_preds == y_val)

    print(f"Validation Log Loss: {val_loss:.15f}")
    print(f"Validation Accuracy: {accuracy:.15f}")

    # 5. Submission Generation
    print("\n[5/5] Generating Submission...")
    test_probs = model.predict_proba(X_test)

    # Create DataFrame
    # Columns: id, Class1, Class2, ... (classes_ is sorted alphabetically by LabelEncoder)
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)
    submission_df.insert(0, ID_COL, test_ids)

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")

    return val_loss
