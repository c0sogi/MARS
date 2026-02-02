import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from library.config import NUMERIC_TYPE


class LinearizedOASClassifier(BaseEstimator, ClassifierMixin):
    """
    Geometrically-Consistent Linearized OAS Discriminant.

    This classifier implements a robust Linear Discriminant Analysis variant designed for
    numerical stability and high precision. It decouples the residual calculation from
    the covariance estimator to enforce geometric consistency and uses a linearized
    formulation for inference to eliminate redundant quadratic terms.
    """

    def __init__(self):
        """
        Initialize the classifier.
        """
        pass

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Args:
            X (array-like): Training data of shape (n_samples, n_features).
            y (array-like): Target values of shape (n_samples,).

        Returns:
            self: Returns the instance itself.
        """
        # 1. Input Validation
        # We accept float32 inputs to maintain quantization regularization (Cite solution_lesson_node_00049)
        X, y = check_X_y(X, y, dtype=[np.float64, np.float32])

        # 2. Encode Labels
        self.le_ = LabelEncoder()
        y_ind = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # 3. Compute Class Centroids and Priors
        # Compute in input precision (float32) to match data quantization
        self.means_ = np.zeros((n_classes, n_features), dtype=X.dtype)
        self.priors_ = np.zeros(n_classes, dtype=X.dtype)

        # Matrix to hold geometrically consistent residuals
        # R = X - mu_y
        R = np.zeros_like(X, dtype=X.dtype)

        for k in range(n_classes):
            # Identify samples belonging to class k
            mask = y_ind == k
            X_k = X[mask]

            if len(X_k) > 0:
                # Compute centroid for class k
                self.means_[k] = np.mean(X_k, axis=0)
                # Compute prior probability
                self.priors_[k] = len(X_k) / n_samples
                # Compute residuals: subtract class centroid from class samples
                R[mask] = X_k - self.means_[k]
            else:
                # Fallback for empty classes (unlikely in this dataset)
                self.means_[k] = 0.0
                self.priors_[k] = 0.0

        # 4. Estimate Covariance using OAS
        # We pass the manually centered residuals R and set assume_centered=True.
        # This prevents OAS from re-centering based on the global mean of R,
        # ensuring the estimator respects the within-class geometry we defined.
        self.covariance_estimator_ = OAS(assume_centered=True)
        self.covariance_estimator_.fit(R)

        # 5. Extract Precision Matrix (Sigma^-1)
        # OAS.precision_ provides the inverse covariance matrix.
        self.precision_ = self.covariance_estimator_.precision_.astype(X.dtype)

        # 6. Derive Linear Weights and Biases
        # The discriminant function is:
        # delta_k(x) = x.T @ Sigma^-1 @ mu_k - 0.5 * mu_k.T @ Sigma^-1 @ mu_k + log(pi_k)

        # Linear Weights (W): W_k = Sigma^-1 @ mu_k
        # We store W such that Z = X @ W.T + b
        # Since Sigma is symmetric, W rows are (Sigma^-1 @ mu_k).T = mu_k.T @ Sigma^-1
        self.W_ = np.dot(self.means_, self.precision_)

        # Bias Term (b): b_k = -0.5 * (mu_k.T @ Sigma^-1 @ mu_k) + log(pi_k)
        # The quadratic term is the dot product of the mean and its projected weight
        quadratic_term = 0.5 * np.sum(self.means_ * self.W_, axis=1)

        # Add epsilon to priors for numerical safety
        safe_priors = np.maximum(self.priors_, 1e-15)
        self.b_ = -quadratic_term + np.log(safe_priors)

        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test data.

        Args:
            X (array-like): Test data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities of shape (n_samples, n_classes).
        """
        check_is_fitted(self)
        X = check_array(X)

        # Upcast everything to float64 for inference to avoid machine epsilon floors (Cite solution_lesson_node_00057)
        X = X.astype(NUMERIC_TYPE)
        W = self.W_.astype(NUMERIC_TYPE)
        b = self.b_.astype(NUMERIC_TYPE)

        # 1. Linear Projection
        # Calculate logits: Z = X @ W.T + b
        logits = np.dot(X, W.T) + b

        # 2. Stable Softmax
        # Subtract max logit for numerical stability to prevent overflow
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return proba.astype(NUMERIC_TYPE)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (array-like): Test data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
