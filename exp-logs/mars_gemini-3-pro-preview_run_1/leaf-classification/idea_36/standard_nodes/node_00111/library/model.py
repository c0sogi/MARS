import numpy as np
from sklearn.preprocessing import LabelEncoder

from library.config import FLOAT_PRECISION
from library.utils import cholesky_solve, stable_softmax


class AnalyticalOASLDA:
    """
    Linear Discriminant Analysis with Analytical Oracle Approximating Shrinkage (OAS).

    Instead of expensive Grid Search, this model uses a closed-form analytical
    formula to estimate the optimal shrinkage parameter alpha for the
    covariance matrix, as derived by Chen et al. (2010).
    """

    def __init__(self):
        self.classes_ = None
        self.le_ = None
        self.W_ = None  # Weight matrix (n_classes, n_features)
        self.b_ = None  # Bias vector (n_classes,)
        self.best_alpha_ = None

    def _compute_stats(self, X, y):
        """
        Computes class priors, class means, and pooled empirical covariance.
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        # Initialize stats
        means = np.zeros((n_classes, n_features), dtype=FLOAT_PRECISION)
        priors = np.zeros(n_classes, dtype=FLOAT_PRECISION)

        # Compute means and priors
        for k, c in enumerate(classes):
            X_c = X[y == c]
            means[k, :] = np.mean(X_c, axis=0, dtype=FLOAT_PRECISION)
            priors[k] = len(X_c) / n_samples

        # Compute pooled covariance
        # S_emp = (1 / (N - K)) * sum((x - mu_k)(x - mu_k)^T)
        X_centered = np.zeros_like(X, dtype=FLOAT_PRECISION)
        for k, c in enumerate(classes):
            mask = y == c
            X_centered[mask] = X[mask] - means[k]

        # Degrees of freedom for pooled covariance: N - K
        dof = n_samples - n_classes
        cov_emp = np.dot(X_centered.T, X_centered) / dof

        return means, priors, cov_emp, dof

    def _compute_oas_alpha(self, cov_emp, n_samples_eff):
        """
        Computes the optimal shrinkage parameter alpha using the OAS formula.

        alpha = ((1 - 2/p) * tr(S^2) + tr^2(S)) /
                ((n + 1 - 2/p) * (tr(S^2) - tr^2(S)/p))
        """
        p = cov_emp.shape[0]
        n = n_samples_eff

        trace_S = np.trace(cov_emp)
        # Efficient calculation of tr(S^2) for symmetric matrix S: sum(S_ij^2)
        trace_S2 = np.sum(cov_emp**2)

        # Numerator
        numerator = (1.0 - 2.0 / p) * trace_S2 + (trace_S**2)

        # Denominator
        denominator = (n + 1.0 - 2.0 / p) * (trace_S2 - (trace_S**2) / p)

        if denominator == 0:
            return 0.0

        alpha = numerator / denominator

        # Clip alpha to [0, 1]
        return min(max(alpha, 0.0), 1.0)

    def _solve_parameters(self, means, priors, covariance):
        """
        Solves for LDA weights W and bias b given statistics and a covariance matrix.
        """
        # Solve Sigma * W^T = mu^T  => W^T = Sigma^-1 * mu^T
        target = means.T
        W_T = cholesky_solve(covariance, target)
        W = W_T.T

        # Compute bias terms: -0.5 * diag(mu * Sigma^-1 * mu^T) + log(priors)
        term1 = -0.5 * np.sum(means * W, axis=1)
        term2 = np.log(priors)
        b = term1 + term2

        return W, b

    def fit(self, X, y):
        """
        Fits the model using Analytical OAS to determine shrinkage.
        """
        X = X.astype(FLOAT_PRECISION)

        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        # Compute Statistics (Means, Priors, Pooled Covariance)
        means, priors, cov_emp, dof = self._compute_stats(X, y_encoded)

        # Compute Analytical Shrinkage Alpha
        self.best_alpha_ = self._compute_oas_alpha(cov_emp, dof)
        print(f"Analytical OAS Optimal Alpha: {self.best_alpha_:.6f}")

        # Compute Shrunk Covariance
        # Sigma(alpha) = (1 - alpha) * S_emp + alpha * (Tr(S_emp) / P) * I
        n_features = cov_emp.shape[0]
        trace = np.trace(cov_emp)
        target = (trace / n_features) * np.eye(n_features, dtype=FLOAT_PRECISION)

        cov_shrunk = (1 - self.best_alpha_) * cov_emp + self.best_alpha_ * target

        # Solve for Weight Matrix and Bias
        self.W_, self.b_ = self._solve_parameters(means, priors, cov_shrunk)

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for samples in X.
        """
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")

        X = X.astype(FLOAT_PRECISION)

        # Linear Inference: Z = X * W^T + b
        logits = np.dot(X, self.W_.T) + self.b_

        # Softmax
        probs = stable_softmax(logits)

        return probs
