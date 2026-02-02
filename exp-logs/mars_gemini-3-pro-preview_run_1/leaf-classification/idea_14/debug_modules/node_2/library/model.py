import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import softmax

from library.config import FLOAT_PRECISION, SEED, SUBMISSION_PATH
from library.utils import compute_log_loss, save_submission
from library.preprocessing import get_preprocessed_data


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis with Oracle Approximating Shrinkage (OAS)
    for covariance estimation.

    This classifier assumes that the data for each class is drawn from a
    multivariate Gaussian distribution with a class-specific mean vector
    but a shared covariance matrix. The covariance matrix is estimated
    using the OAS estimator on the pooled within-class residuals, which
    is robust for high-dimensional, small-sample data.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fit the OAS Discriminant model.

        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        # Validate inputs
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Map classes to indices
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])

        # Compute empirical priors and means
        self.priors_ = np.zeros(n_classes, dtype=FLOAT_PRECISION)
        self.means_ = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)

        # We need centered data for covariance estimation
        X_centered = np.zeros_like(X, dtype=FLOAT_PRECISION)

        for i in range(n_classes):
            mask = y_idx == i
            X_class = X[mask]
            count = X_class.shape[0]

            # Empirical Prior
            self.priors_[i] = count / float(len(y))

            # Empirical Mean
            mean = np.mean(X_class, axis=0)
            self.means_[i] = mean

            # Center the data
            X_centered[mask] = X_class - mean

        # Estimate shared covariance using OAS on pooled residuals
        # assume_centered=True because we manually centered X
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)

        self.covariance_ = oas.covariance_.astype(FLOAT_PRECISION)
        self.precision_ = oas.precision_.astype(FLOAT_PRECISION)

        # Precompute weights and biases for linear decision function
        # Weight matrix W: (n_features, n_classes)
        # W = Sigma^-1 * mu.T
        self.coef_ = np.dot(self.means_, self.precision_)  # (n_classes, n_features)

        # Bias vector b: (n_classes,)
        # b = -0.5 * diag(mu * Sigma^-1 * mu.T) + log(prior)
        # Note: self.coef_ is (mu * Sigma^-1)
        # So we need row-wise dot product of self.coef_ and self.means_
        term1 = -0.5 * np.sum(self.coef_ * self.means_, axis=1)
        term2 = np.log(self.priors_)
        self.intercept_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Estimate probability.

        Args:
            X (array-like): Input data, shape (n_samples, n_features).

        Returns:
            C (array-like): Estimated probabilities, shape (n_samples, n_classes).
        """
        check_is_fitted(self)
        X = check_array(X)
        X = X.astype(FLOAT_PRECISION)

        # Linear decision function: X * W.T + b
        # X: (n_samples, n_features)
        # self.coef_: (n_classes, n_features)
        # self.intercept_: (n_classes,)
        scores = np.dot(X, self.coef_.T) + self.intercept_

        # Apply softmax to get probabilities
        probas = softmax(scores, axis=1)

        return probas.astype(FLOAT_PRECISION)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (array-like): Input data.

        Returns:
            y_pred (array-like): Predicted class labels.
        """
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.classes_[indices]


def train_and_predict(load_cached_data=True):
    """
    Orchestrates the training process:
    1. Loads preprocessed (Gaussianized) data.
    2. Trains the OASDiscriminant model.
    3. Evaluates on the validation set.
    4. Generates predictions for the test set.
    5. Saves the submission file.
    """
    print("Loading data...")
    # Load data using the preprocessing pipeline (Iterative Gaussianization)
    (train_data, val_data, test_data) = get_preprocessed_data(
        load_cached_data=load_cached_data
    )

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, ids_test = test_data

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # Initialize Model
    print("Initializing OASDiscriminant model...")
    model = OASDiscriminant()

    # Fit Model
    print("Fitting model on training set...")
    model.fit(X_train, y_train)

    # Validate
    print("Evaluating on validation set...")
    y_val_pred = model.predict_proba(X_val)
    val_loss = compute_log_loss(y_val, y_val_pred, model.classes_)

    print("-" * 30)
    print(f"Validation Multi-class Log Loss: {val_loss}")
    print("-" * 30)

    # Predict on Test
    print("Generating predictions for test set...")
    y_test_pred = model.predict_proba(X_test)

    # Save Submission
    save_submission(ids_test, y_test_pred, model.classes_, filename=SUBMISSION_PATH)
    print("Process completed successfully.")
