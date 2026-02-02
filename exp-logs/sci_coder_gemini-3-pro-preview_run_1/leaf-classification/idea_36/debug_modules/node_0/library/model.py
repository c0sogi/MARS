import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from library.config import FLOAT_PRECISION, SEED, GRID_SIZE, CV_FOLDS, ALPHA_BOUNDS
from library.utils import cholesky_solve, stable_softmax, clip_probabilities


class MetricOptimizedCholeskyLDA:
    """
    Custom Linear Discriminant Analysis with Metric-Optimized Shrinkage.

    Uses an exact Cholesky solver for inference and optimizes the shrinkage
    parameter alpha via Grid Search to minimize Multi-class Log Loss.
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
        # We compute centered data first
        X_centered = np.zeros_like(X, dtype=FLOAT_PRECISION)
        for k, c in enumerate(classes):
            mask = y == c
            X_centered[mask] = X[mask] - means[k]

        # Compute covariance matrix
        # Note: Using (N - K) as divisor for unbiased estimator in LDA
        cov_emp = np.dot(X_centered.T, X_centered) / (n_samples - n_classes)

        return means, priors, cov_emp

    def _solve_parameters(self, means, priors, covariance):
        """
        Solves for LDA weights W and bias b given statistics and a covariance matrix.

        W = mu * Sigma^-1
        b = -0.5 * diag(mu * Sigma^-1 * mu^T) + log(priors)
        """
        n_features = covariance.shape[0]
        n_classes = means.shape[0]

        # Solve Sigma * W^T = mu^T  => W^T = Sigma^-1 * mu^T
        # We use Cholesky solve for numerical stability
        # Target shape: (n_features, n_classes)
        target = means.T

        # W_T shape: (n_features, n_classes)
        W_T = cholesky_solve(covariance, target)

        # W shape: (n_classes, n_features)
        W = W_T.T

        # Compute bias terms
        # Term 1: -0.5 * mu^T * Sigma^-1 * mu
        # We can compute this efficiently using the already solved W
        # diag(means @ W.T)
        term1 = -0.5 * np.sum(means * W, axis=1)

        # Term 2: log(priors)
        term2 = np.log(priors)

        b = term1 + term2

        return W, b

    def _get_shrunk_covariance(self, emp_cov, alpha):
        """
        Computes the shrunk covariance matrix:
        Sigma(alpha) = (1 - alpha) * S_emp + alpha * (Tr(S_emp) / P) * I
        """
        n_features = emp_cov.shape[0]
        trace = np.trace(emp_cov)
        target = (trace / n_features) * np.eye(n_features, dtype=FLOAT_PRECISION)

        shrunk_cov = (1 - alpha) * emp_cov + alpha * target
        return shrunk_cov

    def fit(self, X, y):
        """
        Fits the model using Stratified K-Fold CV to optimize shrinkage alpha.
        """
        # Ensure float64
        X = X.astype(FLOAT_PRECISION)

        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        # Define Grid
        alphas = np.linspace(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1], GRID_SIZE)

        # Initialize CV
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

        avg_losses = []

        print(
            f"Starting Grid Search for optimal shrinkage (Grid Size: {GRID_SIZE}, Folds: {CV_FOLDS})..."
        )

        # Pre-compute indices to avoid re-splitting
        splits = list(skf.split(X, y_encoded))

        for alpha in alphas:
            fold_losses = []

            for train_idx, val_idx in splits:
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]

                # Compute stats for this fold
                means_f, priors_f, cov_emp_f = self._compute_stats(
                    X_train_fold, y_train_fold
                )

                # Apply shrinkage
                cov_shrunk = self._get_shrunk_covariance(cov_emp_f, alpha)

                # Solve weights
                try:
                    W_f, b_f = self._solve_parameters(means_f, priors_f, cov_shrunk)

                    # Predict
                    logits = np.dot(X_val_fold, W_f.T) + b_f
                    probs = stable_softmax(logits)

                    # Clip for metric calculation stability
                    probs_clipped = clip_probabilities(probs)

                    loss = log_loss(
                        y_val_fold, probs_clipped, labels=np.arange(len(self.classes_))
                    )
                    fold_losses.append(loss)
                except np.linalg.LinAlgError:
                    # In case alpha is too small and matrix is singular (unlikely with shrinkage)
                    fold_losses.append(np.inf)

            avg_loss = np.mean(fold_losses)
            avg_losses.append(avg_loss)

        # Select best alpha
        best_idx = np.argmin(avg_losses)
        self.best_alpha_ = alphas[best_idx]
        best_loss = avg_losses[best_idx]

        print(
            f"Optimal Alpha found: {self.best_alpha_:.6f} with CV Log Loss: {best_loss:.6f}"
        )

        # Refit on full dataset
        print("Refitting on full training set...")
        means_full, priors_full, cov_emp_full = self._compute_stats(X, y_encoded)
        cov_final = self._get_shrunk_covariance(cov_emp_full, self.best_alpha_)

        self.W_, self.b_ = self._solve_parameters(means_full, priors_full, cov_final)

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
