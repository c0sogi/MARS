import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from scipy.special import softmax

import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.preprocessing as preprocessing

# Ensure deterministic behavior
utils.set_seed(config.SEED)


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS).

    This model implements an exact analytical inference pipeline:
    1. Computes empirical class means and priors.
    2. Estimates a robust covariance matrix using OAS on centered residuals.
    3. Derives linear decision boundaries (Weights and Bias) from the precision matrix.
    4. Performs inference using dot products and softmax in float64 precision.
    """

    def __init__(self, assume_centered=True):
        """
        Args:
            assume_centered (bool): If True, data will not be centered before covariance
                                    estimation (we manually center using class means).
        """
        self.assume_centered = assume_centered
        self.classes_ = None
        self.W_ = None  # Weights: (n_classes, n_features)
        self.b_ = None  # Bias: (n_classes,)

    def fit(self, X, y):
        """
        Fits the OAS Discriminant model.

        Args:
            X (array-like): Training features of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,).

        Returns:
            self
        """
        # Enforce high precision
        X = np.array(X, dtype=config.FLOAT_PRECISION)

        # Identify unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Empirical Class Means and Priors
        means = np.zeros((n_classes, n_features), dtype=config.FLOAT_PRECISION)
        priors = np.zeros(n_classes, dtype=config.FLOAT_PRECISION)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            means[i] = np.mean(X_c, axis=0)
            priors[i] = len(X_c) / len(X)

        # 2. Compute Residuals (Centering)
        # Map y to indices 0..K-1 to select correct means
        # Note: y is expected to be integer encoded (0 to n_classes-1)
        X_centered = X - means[y]

        # 3. Estimate Covariance using OAS
        # We use the residuals to estimate the common covariance matrix
        oas = OAS(assume_centered=self.assume_centered)
        oas.fit(X_centered)

        # Extract Precision Matrix (Inverse Covariance)
        precision = oas.precision_.astype(config.FLOAT_PRECISION)

        # 4. Derive Linear Decision Boundaries
        # Weights: W = P * mu^T -> Transposed for calculation: W = mu * P
        # Shape: (n_classes, n_features)
        self.W_ = np.dot(means, precision)

        # Bias: b = -0.5 * diag(mu * P * mu^T) + log(prior)
        # We can compute the quadratic term efficiently: sum(W * mu, axis=1)
        # W[i] is (P * mu_i), so W[i] * mu_i is mu_i^T * P * mu_i
        quad_term = 0.5 * np.sum(self.W_ * means, axis=1)
        self.b_ = -quad_term + np.log(priors)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (array-like): Features of shape (n_samples, n_features).

        Returns:
            array-like: Probabilities of shape (n_samples, n_classes).
        """
        X = np.array(X, dtype=config.FLOAT_PRECISION)

        # Linear Scoring: Z = X W^T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Softmax normalization
        return softmax(logits, axis=1)

    def predict(self, X):
        """
        Predict class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def train_and_predict(load_cached_data=True, limit=None):
    """
    Main execution pipeline:
    1. Loads and fuses data (Tabular + Geometric).
    2. Preprocesses data (Sanitization -> Transformation -> Scaling).
    3. Trains OASDiscriminant.
    4. Validates and prints Log Loss.
    5. Generates Test predictions and saves submission.
    """
    print("Initializing OAS Discriminant Pipeline...")

    # 1. Load Dataframes to get IDs and structure
    # We need the IDs for the submission file
    train_df, val_df, test_df = data_loader.load_datasets(
        load_cached_data=load_cached_data, limit=limit
    )

    # Extract Test IDs for submission
    test_ids = test_df[config.ID_COL].values

    # 2. Get Preprocessed Feature Matrices (Numpy Arrays)
    # This handles the VarianceThreshold -> PowerTransform -> StandardScaler pipeline
    X_train, y_train, X_val, y_val, X_test, classes = (
        preprocessing.get_preprocessed_data(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # 3. Train Model
    print("Fitting OASDiscriminant model...")
    model = OASDiscriminant(assume_centered=True)
    model.fit(X_train, y_train)

    # 4. Validation
    print("Evaluating on Validation set...")
    val_probs = model.predict_proba(X_val)

    # Compute Log Loss
    # y_val are integer indices. We pass labels=range(n_classes) to ensure correct mapping.
    val_loss = utils.compute_log_loss(y_val, val_probs, classes=np.arange(len(classes)))
    print(f"Validation Multi-class Log Loss: {val_loss}")

    # 5. Inference on Test Set
    print("Generating predictions for Test set...")
    test_probs = model.predict_proba(X_test)

    # 6. Create Submission File
    print("Saving submission...")

    # Create DataFrame
    # Columns: id, [class_names...]
    submission_df = pd.DataFrame(test_probs, columns=classes)
    submission_df.insert(0, config.ID_COL, test_ids)

    # Save
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
