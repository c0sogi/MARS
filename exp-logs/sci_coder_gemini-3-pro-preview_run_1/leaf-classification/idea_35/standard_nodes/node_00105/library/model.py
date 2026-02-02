import os
import numpy as np
import pandas as pd
from scipy import linalg, special
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from library.config import OAS_PARAMS, SUBMISSION_PATH, SEED
from library.preprocessing import get_preprocessed_data


class CholeskyOASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    A Linear Discriminant Classifier that uses the OAS estimator for covariance
    and solves for weights using exact Cholesky decomposition.

    This architecture avoids the numerical instability of explicit matrix inversion
    (precision matrix) and the spectral truncation of SVD, providing microscopic
    signal preservation required for high-precision tasks.

    Attributes:
        classes_ (np.ndarray): Unique class labels.
        W_ (np.ndarray): Weight matrix of shape (n_features, n_classes).
        b_ (np.ndarray): Bias vector of shape (n_classes,).
        covariance_ (np.ndarray): Estimated covariance matrix.
    """

    def __init__(self):
        self.classes_ = None
        self.W_ = None
        self.b_ = None
        self.covariance_ = None
        self.le_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation and Cholesky solving.

        Args:
            X (array-like): Training features. Must be float64.
            y (array-like): Training labels.

        Returns:
            self
        """
        # Enforce float64 for precision
        X = np.array(X, dtype=np.float64)

        # Encode labels
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # 1. Compute Empirical Means and Priors
        # We use arithmetic mean as per Lesson 00066
        means = np.zeros((n_classes, n_features), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)

        for k in range(n_classes):
            mask = y_enc == k
            X_k = X[mask]
            means[k] = np.mean(X_k, axis=0)
            priors[k] = len(X_k) / n_samples

        # 2. Compute Centered Residuals
        # R = X - mu_y
        residuals = X - means[y_enc]

        # 3. Estimate Covariance with OAS
        # assume_centered=True is critical as we manually centered residuals
        oas = OAS(**OAS_PARAMS)
        oas.fit(residuals)
        self.covariance_ = oas.covariance_

        # 4. Solve for Weights using Cholesky Decomposition
        # We want to solve Sigma * W = means.T for W.
        # This corresponds to W_k = Sigma^-1 * mu_k
        # Factorize Sigma = L * L.T
        try:
            c, lower = linalg.cho_factor(self.covariance_, lower=True)
        except linalg.LinAlgError:
            # Fallback for extremely rare non-PD cases (though OAS usually prevents this)
            # Add minute jitter
            self.covariance_.flat[:: n_features + 1] += 1e-12
            c, lower = linalg.cho_factor(self.covariance_, lower=True)

        # Solve system
        self.W_ = linalg.cho_solve((c, lower), means.T)

        # 5. Compute Bias Terms
        # b_k = -0.5 * (mu_k . W_k) + log(pi_k)
        # The dot product mu_k . W_k is the diagonal of (means @ W)
        # We compute it efficiently using element-wise multiplication and summation
        # means.T is (n_features, n_classes), W_ is (n_features, n_classes)
        dot_products = np.sum(means.T * self.W_, axis=0)
        self.b_ = -0.5 * dot_products + np.log(priors)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linearized discriminant function.

        Args:
            X (array-like): Input features.

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        X = np.array(X, dtype=np.float64)

        # Linear Projection: Z = XW + b
        logits = X @ self.W_ + self.b_

        # Apply Softmax in float64
        return special.softmax(logits, axis=1)

    def predict(self, X):
        """
        Predicts class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def train_and_predict(load_cached_data=True):
    """
    Executes the full training and prediction pipeline.

    1. Loads preprocessed data (float64, alphanumeric order).
    2. Trains the CholeskyOASDiscriminant.
    3. Evaluates on Validation set (Log Loss).
    4. Generates predictions for Test set.
    5. Saves submission file.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    print("Starting Aligned Exact-Cholesky OAS Pipeline...")

    # 1. Load Data
    # Data is already scaled and power-transformed by the preprocessing module
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = get_preprocessed_data(
        load_cached_data
    )

    print(f"Data Loaded. Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # 2. Train Model
    print("Fitting CholeskyOASDiscriminant...")
    model = CholeskyOASDiscriminant()
    model.fit(X_train, y_train)

    # 3. Validate
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Calculate Log Loss
    # We use the model's classes to ensure alignment.
    # Note: sklearn log_loss handles clipping internally (eps=1e-15 default)
    score = log_loss(y_val, val_probs, labels=model.classes_)
    print(f"Validation Multi-class Log Loss: {score:.15f}")

    # 4. Test Prediction
    print("Generating Test Predictions...")
    test_probs = model.predict_proba(X_test)

    # 5. Create Submission
    # Ensure columns are in the correct order (alphanumeric sort of species)
    # The data loader provides 'classes' which is sorted.
    # The model.classes_ comes from LabelEncoder on y_train, which is also sorted.
    # We verify alignment implicitly.

    submission_df = pd.DataFrame(test_probs, columns=model.classes_)
    submission_df.insert(0, "id", test_ids)

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

    return score
