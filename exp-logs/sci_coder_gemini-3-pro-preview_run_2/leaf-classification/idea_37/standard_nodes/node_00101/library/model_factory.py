import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.covariance import OAS
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from library.config import Config


class OAS_LDA(BaseEstimator, ClassifierMixin):
    """
    Linear Discriminant Analysis with Oracle Approximating Shrinkage (OAS)
    for covariance estimation.

    This estimator computes the pooled covariance matrix using OAS on the
    class-centered data (residuals), then applies the standard LDA decision
    rule. This provides a robust alternative to empirical covariance or
    fixed shrinkage, particularly for high-dimensional, low-sample data.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fits the LDA model using OAS covariance estimation.
        """
        # Ensure inputs are valid and float64 for precision
        X, y = check_X_y(X, y, dtype=Config.NP_DTYPE)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize statistics
        self.means_ = np.zeros((n_classes, n_features), dtype=Config.NP_DTYPE)
        self.priors_ = np.zeros(n_classes, dtype=Config.NP_DTYPE)

        # Center data to compute pooled covariance
        # X_centered = X - mu_y
        X_centered = np.zeros_like(X, dtype=Config.NP_DTYPE)

        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_cls = X[mask]

            # Compute priors and means
            self.priors_[idx] = len(X_cls) / len(X)
            self.means_[idx] = np.mean(X_cls, axis=0)

            # Center the data for this class
            X_centered[mask] = X_cls - self.means_[idx]

        # Estimate Pooled Covariance using OAS
        # OAS assumes data is centered if we want pure covariance of residuals
        # sklearn's OAS will re-center by default, which is fine since mean(X_centered) ~ 0
        oas = OAS()
        oas.fit(X_centered)
        self.covariance_ = oas.covariance_.astype(Config.NP_DTYPE)

        # Compute Precision Matrix (Inverse Covariance)
        # Using pinv for numerical stability, though OAS is generally well-conditioned
        self.precision_ = np.linalg.pinv(self.covariance_)

        # Precompute Linear Coefficients (Weights)
        # coef = means * precision (Shape: n_classes x n_features)
        self.coef_ = np.dot(self.means_, self.precision_)

        # Precompute Intercepts (Bias)
        # intercept_k = -0.5 * (mu_k^T * Sigma^-1 * mu_k) + log(prior_k)
        # The quadratic term can be computed as row-wise dot product of coef and means
        term1 = -0.5 * np.sum(self.coef_ * self.means_, axis=1)
        term2 = np.log(self.priors_)
        self.intercept_ = term1 + term2

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities using the LDA decision function.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=Config.NP_DTYPE)

        # Compute Linear Discriminant Scores
        # scores = X * coef^T + intercept
        # Shape: (n_samples, n_classes)
        scores = np.dot(X, self.coef_.T) + self.intercept_

        # Apply Softmax with numerical stability
        # Shift scores by max to prevent overflow in exp
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs

    def predict(self, X):
        """
        Predicts class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


class ModelFactory:
    """
    Factory class to generate the library of probabilistic experts for the
    Constrained-Basis Dual-Stream Generative Ensemble.
    """

    @staticmethod
    def generate_expert_library():
        """
        Generates a list of expert configurations.

        Each expert is a dictionary containing:
        - 'id': Unique identifier string.
        - 'model': The initialized sklearn-compatible estimator.
        - 'stream': The data stream to use ('stream_a' or 'stream_b').

        Returns:
            list[dict]: List of expert definitions.
        """
        experts = []
        streams = ["stream_a", "stream_b"]

        # 1. Generative Experts (LDA variants)
        # We apply LDA with OAS and Fixed Shrinkage to both streams.
        # Stream A: Parametric (Yeo-Johnson)
        # Stream B: Constrained Non-Parametric (Quantile)

        for stream in streams:
            # A. OAS Shrinkage (Custom Implementation)
            experts.append(
                {"id": f"LDA_OAS_{stream}", "model": OAS_LDA(), "stream": stream}
            )

            # B. Fixed Shrinkage Values (Standard LDA)
            for shrinkage in Config.LDA_SHRINKAGE_VALUES:
                # solver='lsqr' is required for shrinkage in sklearn's LDA
                model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
                experts.append(
                    {
                        "id": f"LDA_Fixed_{shrinkage}_{stream}",
                        "model": model,
                        "stream": stream,
                    }
                )

        # 2. Discriminative Backup (Logistic Regression)
        # Applied only to Stream A (Parametric) as a robust fallback.
        lr_model = LogisticRegressionCV(
            Cs=Config.LR_CS,
            cv=Config.LR_CV_FOLDS,
            scoring=Config.LR_SCORING,
            solver=Config.LR_SOLVER,
            max_iter=Config.LR_MAX_ITER,
            n_jobs=Config.N_JOBS,
            random_state=Config.RANDOM_STATE,
        )

        experts.append(
            {"id": "LogReg_StreamA", "model": lr_model, "stream": "stream_a"}
        )

        return experts
