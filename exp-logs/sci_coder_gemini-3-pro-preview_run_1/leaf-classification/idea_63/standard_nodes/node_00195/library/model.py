import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from scipy.special import softmax

from library.config import SUBMISSION_PATH, SEED
from library.utils import set_seed

# Ensure reproducibility
set_seed(SEED)


class OASDiscriminant:
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS).

    Implements the analytic solution for LDA designed for high-precision float64 inference:
    1. Estimates class means and priors.
    2. Estimates a shared covariance matrix using OAS on centered residuals.
    3. Computes linear decision boundaries (Weights W, Bias b) using the precision matrix.
    4. Performs inference using exact linear algebra.
    """

    def __init__(self):
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.precision_ = None
        self.W_ = None
        self.b_ = None
        self.le_ = None

    def fit(self, X, y):
        """
        Fits the model parameters on the training data.

        Args:
            X (np.ndarray): Training features (float64).
            y (np.ndarray): Training labels.
        """
        # Enforce float64 precision
        X = np.array(X, dtype=np.float64)

        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize statistics
        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        counts = np.zeros(n_classes, dtype=np.float64)

        # Compute Means and Priors
        for k in range(n_classes):
            X_k = X[y_encoded == k]
            counts[k] = X_k.shape[0]
            self.means_[k] = np.mean(X_k, axis=0)

        self.priors_ = counts / counts.sum()

        # Compute Residuals (Centering)
        # R = X - mu_y
        X_means = self.means_[y_encoded]
        residuals = X - X_means

        # Estimate Covariance/Precision using OAS
        # assume_centered=True because we manually subtracted the means to get residuals
        estimator = OAS(assume_centered=True)
        estimator.fit(residuals)

        # Extract Precision Matrix (Inverse Covariance)
        # OAS provides precision_ attribute which is computed via SVD/pinv for stability
        self.precision_ = estimator.precision_.astype(np.float64)

        # Derive Linear Weights and Biases
        # Discriminant function: delta_k(x) = x.T * P * mu_k - 0.5 * mu_k.T * P * mu_k + log(pi_k)
        # We formulate this as: Z = X * W.T + b
        # Where W_k = P * mu_k (stored as row vectors in W_)

        # W shape: (n_classes, n_features)
        # means_: (n_classes, n_features), precision_: (n_features, n_features)
        self.W_ = np.dot(self.means_, self.precision_)

        # b shape: (n_classes,)
        # Quadratic term: 0.5 * diag(means_ @ precision_ @ means_.T)
        # Efficiently computed as row-wise dot product of means_ and W_
        quadratic_term = 0.5 * np.sum(self.means_ * self.W_, axis=1)
        log_priors = np.log(self.priors_)

        self.b_ = log_priors - quadratic_term

        return self

    def predict_proba(self, X):
        """
        Computes class probabilities using the linear discriminant functions.

        Args:
            X (np.ndarray): Features.

        Returns:
            np.ndarray: Probability matrix (n_samples, n_classes).
        """
        X = np.array(X, dtype=np.float64)

        # Linear Logits: Z = X @ W.T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Softmax
        probs = softmax(logits, axis=1)

        # Clipping to avoid log(0) in metric calculation
        # Metric definition: max(min(p, 1-10^-15), 10^-15)
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)

        return probs


def train_model(X_train, y_train, X_val, y_val):
    """
    Trains the OAS Discriminant model and evaluates on validation set.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.

    Returns:
        model: Trained OASDiscriminant instance.
    """
    print("Initializing OAS Discriminant...")
    model = OASDiscriminant()

    print(
        f"Fitting model on {len(X_train)} samples with {X_train.shape[1]} features..."
    )
    model.fit(X_train, y_train)

    print("Evaluating on validation set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # We pass labels to ensure correct mapping between probabilities and ground truth
    score = log_loss(y_val, val_probs, labels=model.classes_)

    print(f"Validation Multi-class Log Loss: {score}")

    return model


def generate_submission(model, X_test, test_ids):
    """
    Generates predictions for the test set and saves to submission file.

    Args:
        model: Trained model.
        X_test: Test features.
        test_ids: Test image IDs.
    """
    print("Generating predictions for test set...")
    test_probs = model.predict_proba(X_test)

    # Create DataFrame
    # Columns must be the class names
    df_sub = pd.DataFrame(test_probs, columns=model.classes_)

    # Insert ID column at the beginning
    df_sub.insert(0, "id", test_ids)

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # Save
    print(f"Saving submission to {SUBMISSION_PATH}...")
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
