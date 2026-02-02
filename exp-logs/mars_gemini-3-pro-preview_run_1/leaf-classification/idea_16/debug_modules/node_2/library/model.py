import numpy as np
from sklearn.covariance import OAS
from scipy.special import logsumexp
from library.utils import set_seed


class FixedMeanOASDiscriminant:
    """
    A custom Linear Discriminant classifier that decouples mean estimation from
    covariance estimation.

    This allows for a semi-supervised 'Fixed-Mean' strategy where:
    1. Class Means are fixed based on labeled training data (preventing drift).
    2. Covariance is estimated using OAS on residuals from both training data
       and pseudo-labeled test data (improving manifold estimation).
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None  # shape (n_classes, n_features)
        self.priors_ = None  # shape (n_classes,)
        self.precision_ = None  # shape (n_features, n_features)

        # Precomputed weights for linear scoring: score = X @ coef_.T + intercept_
        self.coef_ = None  # shape (n_classes, n_features)
        self.intercept_ = None  # shape (n_classes,)

    def fit_means(self, X, y):
        """
        Calculates and stores class centroids and priors from labeled data.

        Args:
            X (array-like): Feature matrix (n_samples, n_features).
            y (array-like): Target labels (n_samples,).

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # Identify unique classes (sorted)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        # Compute mean and prior for each class
        for i, cls in enumerate(self.classes_):
            mask = y == cls
            X_cls = X[mask]
            self.means_[i] = np.mean(X_cls, axis=0)
            self.priors_[i] = len(X_cls) / len(X)

        return self

    def compute_residuals(self, X, y):
        """
        Computes residuals by subtracting the assigned class mean from each sample.
        Used to center data before covariance estimation.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Labels corresponding to X (can be pseudo-labels).

        Returns:
            np.ndarray: Residuals matrix (same shape as X).
        """
        if self.means_ is None:
            raise RuntimeError("Must call fit_means before compute_residuals.")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # Map class labels to indices
        cls_to_idx = {cls: i for i, cls in enumerate(self.classes_)}

        # Create an array of indices corresponding to y
        # We assume y contains valid labels found in self.classes_
        indices = np.array([cls_to_idx[label] for label in y])

        # Retrieve corresponding means
        means_selected = self.means_[indices]

        return X - means_selected

    def fit_covariance(self, residuals):
        """
        Estimates the shared covariance matrix using Oracle Approximating Shrinkage (OAS)
        on the provided residuals.

        Args:
            residuals (array-like): Centered data (X - mu).

        Returns:
            self
        """
        residuals = np.asarray(residuals, dtype=np.float64)

        # Estimate covariance using OAS
        # We use assume_centered=True because residuals are centered by definition
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        self.precision_ = oas.precision_

        # Precompute linear discriminant parameters for fast inference
        # The log-posterior is proportional to:
        # log P(k|x) ~ x.T * (Precision * mu_k) - 0.5 * mu_k.T * Precision * mu_k + log(prior_k)

        # coef_ [k] = (Precision * mu_k).T = mu_k.T * Precision
        self.coef_ = self.means_ @ self.precision_

        # intercept_ [k] = -0.5 * mu_k.T * Precision * mu_k + log(prior_k)
        # Note: (means_ * coef_) gives element-wise product. Sum over axis 1 gives the dot product.
        quadratic_term = -0.5 * np.sum(self.means_ * self.coef_, axis=1)
        log_prior_term = np.log(self.priors_)

        self.intercept_ = quadratic_term + log_prior_term

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for samples in X.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Probability matrix (n_samples, n_classes).
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Must call fit_covariance before predict_proba.")

        X = np.asarray(X, dtype=np.float64)

        # Calculate linear scores (logits)
        # shape: (n_samples, n_classes)
        scores = X @ self.coef_.T + self.intercept_

        # Apply Softmax with numerical stability
        # log_softmax = scores - logsumexp(scores)
        lse = logsumexp(scores, axis=1, keepdims=True)
        log_probs = scores - lse

        return np.exp(log_probs)
