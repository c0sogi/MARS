import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from library.config import FLOAT_PRECISION, SEED
from library.utils import get_logger

logger = get_logger("model")


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    A Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS)
    for robust covariance estimation.

    This implementation explicitly linearizes the decision boundary to avoid
    quadratic instability during inference and enforces float64 precision.

    Attributes:
        classes_ (np.ndarray): The unique class labels.
        le_ (LabelEncoder): Encoder for target labels.
        W_ (np.ndarray): The weight matrix of shape (n_classes, n_features).
        b_ (np.ndarray): The bias vector of shape (n_classes,).
    """

    def __init__(self):
        self.classes_ = None
        self.le_ = None
        self.W_ = None
        self.b_ = None
        # We use assume_centered=True because we manually center the data
        # based on class means before fitting the estimator.
        self.estimator = OAS(assume_centered=True)

    def fit(self, X, y):
        """
        Fits the OAS Discriminant model.

        1. Computes class means and priors.
        2. Centers data (residuals).
        3. Fits OAS on residuals to get the Precision matrix.
        4. Derives linear weights W and bias b.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Training labels.

        Returns:
            self
        """
        logger.info("Fitting OASDiscriminant...")

        # Ensure float64
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        # Encode labels
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        logger.info(
            f"Training on {X.shape[0]} samples, {n_features} features, {n_classes} classes."
        )

        # 1. Compute empirical class means and priors
        means = np.zeros((n_classes, n_features), dtype=np.float64)
        priors = np.zeros(n_classes, dtype=np.float64)

        # We also need centered data (residuals) for OAS
        residuals = np.zeros_like(X, dtype=np.float64)

        for k in range(n_classes):
            mask = y_enc == k
            X_k = X[mask]

            # Prior
            priors[k] = X_k.shape[0] / X.shape[0]

            # Mean
            mu_k = np.mean(X_k, axis=0)
            means[k] = mu_k

            # Residuals for this class
            residuals[mask] = X_k - mu_k

        # 2. Estimate Covariance/Precision using OAS on Residuals
        # OAS minimizes MSE of the covariance estimate.
        logger.info("Estimating covariance with OAS...")
        self.estimator.fit(residuals)

        # Extract Precision Matrix (Inverse Covariance)
        # sklearn OAS provides precision_ attribute after fitting
        P = self.estimator.precision_

        # 3. Derive Linear Weights and Bias
        # W_k = P * mu_k
        # b_k = -0.5 * (mu_k.T * P * mu_k) + log(prior_k)
        # Note: mu_k.T * P * mu_k is equivalent to mu_k . W_k

        logger.info("Deriving linear decision boundaries...")

        # W shape: (n_classes, n_features)
        # P shape: (n_features, n_features)
        # means.T shape: (n_features, n_classes)
        # W = (P @ means.T).T -> means @ P
        self.W_ = np.dot(means, P)

        # b shape: (n_classes,)
        # The quadratic term: 0.5 * diag(means @ W_.T)
        # We can compute row-wise dot product of means and W_
        quadratic_term = 0.5 * np.sum(means * self.W_, axis=1)
        log_priors = np.log(priors)

        self.b_ = log_priors - quadratic_term

        logger.info("Model fitted successfully.")
        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linearized discriminant function.

        Z = X @ W.T + b
        Probabilities = softmax(Z)

        Args:
            X (pd.DataFrame or np.ndarray): Features to predict.

        Returns:
            pd.DataFrame: Probabilities with class names as columns.
        """
        check_is_fitted(self, ["W_", "b_", "classes_"])

        X = np.array(X, dtype=np.float64)

        # 1. Linear Projection (Logits)
        # X: (n_samples, n_features)
        # W_: (n_classes, n_features)
        # b_: (n_classes,)
        # Z: (n_samples, n_classes)
        Z = np.dot(X, self.W_.T) + self.b_

        # 2. Softmax
        # axis=1 ensures sum over classes is 1
        probs = softmax(Z, axis=1)

        # Return as DataFrame for easy submission handling
        return pd.DataFrame(probs, columns=self.classes_)

    def predict(self, X):
        """
        Predicts class labels.

        Args:
            X (pd.DataFrame or np.ndarray): Features to predict.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probs = self.predict_proba(X)
        # Get index of max probability
        max_indices = np.argmax(probs.values, axis=1)
        # Map back to original class labels
        return self.classes_[max_indices]
