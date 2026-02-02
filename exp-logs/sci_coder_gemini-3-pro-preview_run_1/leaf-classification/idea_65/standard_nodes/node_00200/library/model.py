import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.preprocessing import LabelBinarizer
from library import config


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    A Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS).

    This classifier implements the 'Sanitized Parsimonious Geometric High-Precision OAS Discriminant'
    strategy. It explicitly computes class means and priors, estimates a shared covariance
    matrix using OAS on centered residuals, and derives linear decision boundaries.

    All internal computations are strictly enforced to use float64 precision to avoid
    numerical instability in high-dimensional feature spaces.
    """

    def __init__(self):
        """
        Initializes the OASDiscriminant.
        """
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.precision_ = None
        self.coef_ = None
        self.intercept_ = None
        self.covariance_estimator_ = None

    def fit(self, X, y):
        """
        Fits the model according to the given training data.

        1. Computes empirical class means and priors.
        2. Computes residuals (X - class_mean).
        3. Fits OAS covariance estimator on residuals.
        4. Derives linear weights (coef_) and bias (intercept_) using the precision matrix.

        Args:
            X (array-like): Training data of shape (n_samples, n_features).
            y (array-like): Target values of shape (n_samples,).

        Returns:
            self: Returns the instance itself.
        """
        # Enforce float64 precision for input
        X = check_array(X, dtype=config.FLOAT_PRECISION)
        X, y = check_X_y(X, y, dtype=config.FLOAT_PRECISION)

        # Identify unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize statistics
        self.means_ = np.zeros((n_classes, n_features), dtype=config.FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=config.FLOAT_PRECISION)

        # Compute Means and Priors
        # We iterate through classes to compute arithmetic means
        residuals = np.zeros_like(X, dtype=config.FLOAT_PRECISION)

        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_cls = X[mask]

            # Empirical Prior
            self.priors_[idx] = X_cls.shape[0] / X.shape[0]

            # Empirical Mean
            mean_vec = np.mean(X_cls, axis=0)
            self.means_[idx, :] = mean_vec

            # Compute Residuals for Covariance Estimation
            residuals[mask] = X_cls - mean_vec

        # Estimate Covariance using OAS
        # We use assume_centered=True because we have explicitly centered the data (residuals)
        self.covariance_estimator_ = OAS(assume_centered=True)
        self.covariance_estimator_.fit(residuals)

        # Extract Precision Matrix (Inverse Covariance)
        # OAS implementation provides precision_ via SVD-based pseudo-inverse logic
        self.precision_ = self.covariance_estimator_.precision_.astype(
            config.FLOAT_PRECISION
        )

        # Derivation of Linear Decision Boundaries
        # For LDA, the discriminant function is:
        # delta_k(x) = x.T * P * mu_k - 0.5 * mu_k.T * P * mu_k + log(pi_k)
        # This can be rewritten as linear score: x.T * W_k + b_k
        # Where W_k = P * mu_k (vector)
        # And b_k = -0.5 * (mu_k.T * W_k) + log(pi_k)

        # coef_ shape: (n_classes, n_features) -> Rows are weight vectors
        # intercept_ shape: (n_classes,)

        # W = (P @ means.T).T = means @ P (since P is symmetric)
        self.coef_ = np.dot(self.means_, self.precision_)

        # b = -0.5 * diag(means @ W.T) + log(priors)
        # We compute the quadratic term efficiently
        quadratic_term = -0.5 * np.sum(self.means_ * self.coef_, axis=1)
        self.intercept_ = quadratic_term + np.log(self.priors_)

        return self

    def predict_proba(self, X):
        """
        Probability estimation for X.

        Computes linear scores (logits) and applies the softmax function.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities of shape (n_samples, n_classes).
        """
        check_is_fitted(self)
        X = check_array(X, dtype=config.FLOAT_PRECISION)

        # Compute Logits: Z = X @ W.T + b
        logits = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax
        # scipy.special.softmax handles numerical stability (exp-normalize trick)
        proba = softmax(logits, axis=1)

        return proba

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=config.FLOAT_PRECISION)

        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)

        return self.classes_[indices]
