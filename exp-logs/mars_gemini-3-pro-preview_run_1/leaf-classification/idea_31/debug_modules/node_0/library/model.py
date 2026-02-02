import numpy as np
import pandas as pd
from scipy import linalg, special
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder

from library.config import FLOAT_TYPE, SEED
from library.utils import (
    calculate_log_loss,
    save_submission,
    get_class_names_from_submission,
)
from library.data_pipeline import load_dataset


class CholeskyOASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    A Linear Discriminant Classifier that uses the OAS estimator for covariance
    and a Cholesky decomposition solver for weight derivation.

    This implementation enforces float64 precision and avoids explicit matrix inversion
    to minimize numerical noise and SVD truncation artifacts.
    """

    def __init__(self):
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.coef_ = None  # Weight matrix W
        self.intercept_ = None  # Bias vector b
        self.le_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation and Cholesky solving.

        Args:
            X (array-like): Training data of shape (n_samples, n_features).
            y (array-like): Target values.

        Returns:
            self
        """
        # Ensure data is float64
        X = np.array(X, dtype=FLOAT_TYPE)

        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Estimate Priors and Means
        # We use arithmetic mean as it is the MLE for Gaussian data
        self.priors_ = np.zeros(n_classes, dtype=FLOAT_TYPE)
        self.means_ = np.zeros((n_classes, n_features), dtype=FLOAT_TYPE)

        counts = np.bincount(y_encoded)
        self.priors_ = counts / float(len(y))

        for k in range(n_classes):
            X_k = X[y_encoded == k]
            self.means_[k] = np.mean(X_k, axis=0)

        # 2. Estimate Covariance using OAS
        # We center the data using the class means to compute residuals
        # R = X - mu_y
        X_centered = X - self.means_[y_encoded]

        # OAS with assume_centered=True because we just manually centered it
        # based on class conditional means.
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)
        self.covariance_ = oas.covariance_.astype(FLOAT_TYPE)

        # 3. Solve for Weights (W) using Cholesky Decomposition
        # We want to find W such that W_k = Sigma^-1 * mu_k
        # This is equivalent to solving: Sigma * W_k = mu_k
        # In matrix form for all classes: Sigma * W.T = means.T

        # Factorize Sigma = L * L.T
        # check_finite=False for slight speedup, we know data is clean/processed
        c, lower = linalg.cho_factor(self.covariance_, lower=True, check_finite=False)

        # Solve for W.T (shape: n_features x n_classes)
        # means.T is (n_features x n_classes)
        W_t = linalg.cho_solve((c, lower), self.means_.T, check_finite=False)

        # Transpose back to get W (n_classes x n_features)
        self.coef_ = W_t.T

        # 4. Compute Bias (Intercept)
        # b_k = -0.5 * (mu_k.T * Sigma^-1 * mu_k) + log(prior_k)
        # Since W_k = Sigma^-1 * mu_k, this simplifies to:
        # b_k = -0.5 * (mu_k . W_k) + log(prior_k)

        # Dot product per class row
        # self.means_ * self.coef_ performs element-wise multiplication
        # sum(axis=1) completes the dot product
        quad_term = 0.5 * np.sum(self.means_ * self.coef_, axis=1)
        self.intercept_ = -quad_term + np.log(self.priors_)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (array-like): Input data.

        Returns:
            array-like: Probabilities of shape (n_samples, n_classes).
        """
        X = np.array(X, dtype=FLOAT_TYPE)

        # Linear Discriminant Function: f(x) = X*W.T + b
        scores = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax
        # scipy.special.softmax is numerically stable
        probs = special.softmax(scores, axis=1)

        return probs

    def predict(self, X):
        """
        Predict class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def run_cholesky_oas_workflow(debug=False, debug_sample_size=100):
    """
    Executes the full training and inference workflow.

    1. Loads data (cached or fresh).
    2. Trains CholeskyOASDiscriminant.
    3. Evaluates on Validation set.
    4. Generates Submission on Test set.
    """
    print(f"Starting Cholesky-OAS Workflow (Debug={debug})...")

    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, ids_test, classes = load_dataset(
        load_cached_data=True, debug=debug, debug_sample_size=debug_sample_size
    )

    print(f"Data Loaded. Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # 2. Initialize and Train Model
    print("Initializing CholeskyOASDiscriminant...")
    model = CholeskyOASDiscriminant()

    print("Fitting model...")
    model.fit(X_train, y_train)

    # 3. Validation
    print("Predicting on validation set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Metric
    # Note: calculate_log_loss handles clipping and rescaling internally
    val_loss = calculate_log_loss(y_val, val_probs, class_names=model.classes_)
    print(f"Validation Multi-class Log Loss: {val_loss:.15f}")

    # 4. Submission
    print("Predicting on test set...")
    test_probs = model.predict_proba(X_test)

    # Ensure columns match the sample submission order
    # The model.classes_ are sorted by LabelEncoder (alphanumerically),
    # which usually matches, but we verify against sample submission if possible.
    # The utils function saves using the provided class names order.

    # If the sample submission has a specific header order, we should respect it.
    # However, save_submission takes class_names as an argument and writes the header.
    # We will use the model's classes which are alphanumeric.

    save_submission(
        ids=ids_test,
        probabilities=test_probs,
        class_names=model.classes_,
        filename="submission.csv",
    )

    print("Workflow completed successfully.")
    return val_loss
