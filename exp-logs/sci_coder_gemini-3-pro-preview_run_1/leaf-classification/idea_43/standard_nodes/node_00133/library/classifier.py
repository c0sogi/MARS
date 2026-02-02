import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.metrics import accuracy_score, log_loss
from scipy.special import softmax
from library.config import PRECISION_TYPE


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS).

    This classifier implements a linear decision boundary derived from Gaussian
    distributions with a shared covariance matrix. The covariance matrix is
    estimated using OAS on the class-centered residuals to handle high-dimensional
    feature spaces and collinearity.

    Inference is performed using the exact linear algebra formulation:
        Logits = X @ W.T + b
    where W and b are pre-computed in float64 precision.
    """

    def __init__(self, assume_centered=True):
        """
        Args:
            assume_centered (bool): If True, data will not be centered by OAS
                                    (since we manually center residuals).
        """
        self.assume_centered = assume_centered
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.precision_ = None
        self.W_ = None
        self.b_ = None

    def fit(self, X, y):
        """
        Fit the OAS Discriminant model.

        Args:
            X (array-like): Training features of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,).

        Returns:
            self: Returns the instance itself.
        """
        # 1. Enforce Double Precision
        X = np.asarray(X, dtype=PRECISION_TYPE)
        y = np.asarray(y)

        # 2. Extract Class Statistics
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.means_ = np.zeros((n_classes, n_features), dtype=PRECISION_TYPE)
        self.priors_ = np.zeros(n_classes, dtype=PRECISION_TYPE)

        # We need residuals for covariance estimation
        # Residuals = X - ClassMean
        residuals = np.zeros_like(X, dtype=PRECISION_TYPE)

        print(
            f"Fitting OASDiscriminant on {n_samples} samples with {n_features} features..."
        )

        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_cls = X[mask]
            count = X_cls.shape[0]

            # Empirical Mean
            mean_cls = np.mean(X_cls, axis=0)
            self.means_[idx, :] = mean_cls

            # Empirical Prior
            self.priors_[idx] = count / n_samples

            # Compute Residuals
            residuals[mask] = X_cls - mean_cls

        # 3. Estimate Covariance via OAS
        # We use the residuals which are already centered around 0 (conceptually)
        oas = OAS(assume_centered=self.assume_centered)
        oas.fit(residuals)

        # Extract Precision Matrix (Inverse Covariance)
        self.precision_ = oas.precision_.astype(PRECISION_TYPE)

        # 4. Compute Linear Decision Boundaries
        # W = P * mu.T  (Shape: n_features x n_classes) -> Transpose for W (n_classes x n_features)
        # Actually: W_k = P * mu_k.
        # Matrix form: W = (P @ self.means_.T).T = self.means_ @ P.T = self.means_ @ P (since P is symmetric)
        self.W_ = np.dot(self.means_, self.precision_)

        # Bias b_k = -0.5 * (mu_k.T * P * mu_k) + log(pi_k)
        # The quadratic term is the diagonal of (means @ P @ means.T)
        # Or simpler: row-wise dot product of means and W
        quad_term = 0.5 * np.sum(self.means_ * self.W_, axis=1)
        self.b_ = -quad_term + np.log(self.priors_)

        # 5. Training Metrics
        # Compute training score to verify fit
        y_pred_proba = self.predict_proba(X)
        train_loss = log_loss(y, y_pred_proba, labels=self.classes_)
        y_pred = self.classes_[np.argmax(y_pred_proba, axis=1)]
        train_acc = accuracy_score(y, y_pred)

        print(
            f"Training Completed. Accuracy: {train_acc:.6f}, Log Loss: {train_loss:.6f}"
        )

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (array-like): Input features.

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X = np.asarray(X, dtype=PRECISION_TYPE)

        # Linear Projection: Z = X @ W.T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Softmax
        proba = softmax(logits, axis=1)

        return proba

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X (array-like): Input features.

        Returns:
            np.ndarray: Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
