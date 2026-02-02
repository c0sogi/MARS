import numpy as np
from scipy.special import softmax
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from library.config import FLOAT_PRECISION, SEED, OAS_ASSUME_CENTERED


class RobustOASClassifier:
    """
    A Linear Discriminant Classifier that uses Oracle Approximating Shrinkage (OAS)
    for covariance estimation and SVD-based pseudo-inverse for weight calculation.

    This approach utilizes the library-provided precision matrix from OAS, which
    uses SVD (via pinv) to handle ill-conditioned covariance matrices more robustly
    than Cholesky or standard linear solvers in high-dimensional regimes.
    Cite {solution_lesson_node_00062}
    """

    def __init__(self):
        self.classes_ = None
        self.le_ = None
        self.W_ = None  # Weight matrix (n_classes, n_features)
        self.b_ = None  # Bias vector (n_classes,)
        self.covariance_ = None
        self.precision_matrix_ = None

    def fit(self, X, y):
        """
        Fits the model using OAS covariance estimation and SVD-based precision calculation.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Training labels of shape (n_samples,).

        Returns:
            self
        """
        # 1. Enforce Precision
        X = np.array(X, dtype=FLOAT_PRECISION)

        # 2. Encode Labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # 3. Compute Class Means and Priors
        # We compute arithmetic means and priors based on sample counts
        means = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        priors = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        for k in range(n_classes):
            mask = y_encoded == k
            X_k = X[mask]
            means[k] = np.mean(X_k, axis=0)
            priors[k] = len(X_k) / n_samples

        # 4. Compute Residuals for Covariance Estimation
        # R = X - mu_y
        # We subtract the specific class mean from each sample to center the data
        # for the pooled covariance calculation.
        residuals = np.empty_like(X, dtype=FLOAT_PRECISION)
        for k in range(n_classes):
            mask = y_encoded == k
            residuals[mask] = X[mask] - means[k]

        # 5. Estimate Covariance using OAS
        # We use assume_centered=True because we manually centered the residuals
        oas = OAS(assume_centered=OAS_ASSUME_CENTERED)
        oas.fit(residuals)
        self.covariance_ = oas.covariance_.astype(FLOAT_PRECISION)

        # 6. Weight Calculation via SVD-based Precision
        # We use the library-provided precision_ attribute which typically uses SVD-based
        # pseudo-inverse (pinv). This is numerically more stable than solving linear systems
        # or Cholesky decomposition in high-dimensional, correlated regimes.
        # Cite {solution_lesson_node_00062}

        # W^T = Sigma^-1 * Means^T = Precision * Means^T
        self.precision_matrix_ = oas.precision_.astype(FLOAT_PRECISION)
        W_transposed = np.dot(self.precision_matrix_, means.T)

        # Transpose back to get W of shape (n_classes, n_features)
        self.W_ = W_transposed.T

        # 7. Compute Bias Terms
        # b_k = -0.5 * (mu_k . w_k) + log(pi_k)
        dot_products = np.sum(means * self.W_, axis=1)  # (n_classes,)
        self.b_ = -0.5 * dot_products + np.log(priors)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the linearized decision boundaries.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities of shape (n_samples, n_classes).
        """
        # Enforce Precision
        X = np.array(X, dtype=FLOAT_PRECISION)

        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        # Linear Projection: Z = X W^T + b
        # X: (n_samples, n_features)
        # W^T: (n_features, n_classes)
        # b: (n_classes,)
        logits = np.dot(X, self.W_.T) + self.b_

        # Softmax in float64
        probs = softmax(logits, axis=1)

        return probs
