import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from scipy.special import softmax

from library.config import (
    FLOAT_PRECISION,
    RANDOM_SEED,
    SUBMISSION_PATH,
    OAS_ASSUME_CENTERED,
)
from library.data_processing import get_processed_data

# Ensure reproducibility
np.random.seed(RANDOM_SEED)


class HighPrecisionOASDiscriminant:
    """
    A Custom Linear Discriminant Classifier using OAS Covariance Estimation.
    Designed for exact analytical inference using float64 precision.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.covariance_estimator_ = None
        self.precision_ = None
        self.W_ = None
        self.b_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation on class-centered residuals.

        Args:
            X (np.ndarray): Training features (float64).
            y (np.ndarray): Target labels (encoded integers).
        """
        # Ensure float64
        X = X.astype(FLOAT_PRECISION)

        # Identify unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 1. Compute Empirical Class Means and Priors
        self.means_ = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        # We need residuals for covariance estimation
        # R = X - mu_y
        residuals = np.zeros_like(X, dtype=FLOAT_PRECISION)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            count = X_c.shape[0]

            # Arithmetic Mean
            mean_c = np.mean(X_c, axis=0)
            self.means_[idx, :] = mean_c

            # Prior
            self.priors_[idx] = count / X.shape[0]

            # Compute residuals for this class
            residuals[y == c] = X_c - mean_c

        # 2. Estimate Covariance using OAS
        # We use assume_centered=True because we manually centered the data (residuals)
        self.covariance_estimator_ = OAS(assume_centered=OAS_ASSUME_CENTERED)
        self.covariance_estimator_.fit(residuals)

        # 3. Extract Precision Matrix (Inverse Covariance)
        # OAS implementation in sklearn provides precision_ attribute
        self.precision_ = self.covariance_estimator_.precision_.astype(FLOAT_PRECISION)

        # 4. Derive Linear Decision Boundaries (Weights and Bias)
        # W_k = P * mu_k
        # b_k = -0.5 * (mu_k.T * P * mu_k) + log(pi_k)
        # Note: We compute W as a matrix where row k is W_k

        # W shape: (n_classes, n_features)
        self.W_ = np.dot(self.means_, self.precision_)

        # b shape: (n_classes,)
        # Compute quadratic term: diag(means * W.T)
        quadratic_term = -0.5 * np.sum(self.means_ * self.W_, axis=1)
        log_priors = np.log(self.priors_)
        self.b_ = quadratic_term + log_priors

        return self

    def predict_proba(self, X):
        """
        Predicts probabilities using linear scoring and softmax.

        Args:
            X (np.ndarray): Features (float64).

        Returns:
            np.ndarray: Class probabilities.
        """
        X = X.astype(FLOAT_PRECISION)

        # Linear Scoring: Z = X * W.T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Softmax in float64
        probs = softmax(logits, axis=1)

        # Clip probabilities to avoid log loss extremes
        # max(min(p, 1-1e-15), 1e-15)
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1.0 - epsilon)

        return probs


def train_and_predict(load_cached_data=True):
    """
    Main pipeline function to load data, train the model, evaluate, and generate submission.
    """
    print("Loading and processing data...")
    X_train, y_train_raw, X_val, y_val_raw, X_test, test_ids = get_processed_data(
        load_cached_data=load_cached_data
    )

    print(
        f"Data Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    # Encode Labels
    # LabelEncoder sorts classes alphabetically, which aligns with submission format
    le = LabelEncoder()
    # Fit on all possible labels (train + val) to ensure consistency
    all_labels = np.concatenate([y_train_raw, y_val_raw])
    le.fit(all_labels)

    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    # Initialize Model
    print("Initializing HighPrecisionOASDiscriminant...")
    model = HighPrecisionOASDiscriminant()

    # Train
    print("Fitting model...")
    model.fit(X_train, y_train)

    # Evaluate on Validation
    print("Evaluating on Validation set...")
    val_probs = model.predict_proba(X_val)
    val_loss = log_loss(y_val, val_probs)

    print(f"Validation Multi-class Log Loss: {val_loss}")

    # Predict on Test
    print("Generating Test predictions...")
    test_probs = model.predict_proba(X_test)

    # Prepare Submission
    # Columns must be the sorted species names
    species_columns = le.classes_

    submission_df = pd.DataFrame(test_probs, columns=species_columns)
    submission_df.insert(0, "id", test_ids)

    # Save Submission
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")

    return val_loss
