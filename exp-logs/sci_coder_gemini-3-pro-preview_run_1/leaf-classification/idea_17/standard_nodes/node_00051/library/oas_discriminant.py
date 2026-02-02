import numpy as np
from sklearn.covariance import OAS
from scipy.special import softmax
from library.utils import validate_precision


class OASDiscriminant:
    """
    Precision-Quantized Supervised OAS Discriminant.

    This classifier implements a Linear Discriminant Analysis (LDA) variant where:
    1. Class means are estimated strictly from training data.
    2. Covariance is estimated using Oracle Approximating Shrinkage (OAS) on training residuals.
    3. All inference parameters are quantized to float32 to enforce precision consistency
       and allow probability saturation.
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None  # Shape: (n_classes, n_features)
        self.precision_ = None  # Shape: (n_features, n_features)
        self.priors_ = None  # Shape: (n_classes,)
        self.covariance_ = None  # Shape: (n_features, n_features)

    def fit_means(self, X, y):
        """
        Computes class centroids and priors from labeled data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.means_[idx] = np.mean(X_cls, axis=0)
            self.priors_[idx] = len(X_cls) / len(X)

    def compute_residuals(self, X, y_indices):
        """
        Computes residuals by subtracting the corresponding class mean from each sample.

        Args:
            X (np.ndarray): Feature matrix.
            y_indices (np.ndarray): Integer indices corresponding to self.classes_ for each sample.

        Returns:
            np.ndarray: Residuals matrix.
        """
        # Map indices to means
        # y_indices are 0 to n_classes-1
        means_per_sample = self.means_[y_indices]
        return X - means_per_sample

    def fit_covariance(self, residuals):
        """
        Estimates the shared covariance matrix using OAS on the provided residuals.
        Updates self.covariance_ and self.precision_.

        Args:
            residuals (np.ndarray): Centered data.
        """
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        self.covariance_ = oas.covariance_
        self.precision_ = oas.precision_

    def _quantize_parameters(self):
        """
        Casts stored parameters to float32 to enforce precision consistency.
        """
        self.means_ = validate_precision(self.means_, "Class Means", np.float32)
        self.precision_ = validate_precision(
            self.precision_, "Precision Matrix", np.float32
        )
        # Priors can remain higher precision or float32, usually less critical for saturation
        # but for consistency we cast them too.
        self.priors_ = validate_precision(self.priors_, "Priors", np.float32)

    def fit(self, X_train, y_train):
        """
        Fits the model using the Supervised OAS strategy.

        Args:
            X_train (np.ndarray): Training features (float32).
            y_train (np.ndarray): Training labels.
        """
        print("Step 1: Estimating Class Means from Training Data...")
        # Ensure inputs are float64 for stable estimation initially
        X_train_64 = X_train.astype(np.float64)
        self.fit_means(X_train_64, y_train)

        # Map y_train to indices [0, n_classes-1]
        # Assuming y_train contains values from self.classes_
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        y_train_idx = np.array([class_to_idx[yi] for yi in y_train])

        print("Step 2: Computing Training Residuals...")
        residuals_train = self.compute_residuals(X_train_64, y_train_idx)

        print("Step 3: Estimating Covariance (OAS)...")
        self.fit_covariance(residuals_train)

        print("Step 4: Enforcing Precision Consistency (Quantizing to float32)...")
        self._quantize_parameters()

        return self

    def _predict_proba_internal(self, X):
        """
        Internal prediction method. Can handle float64 or float32.
        Calculates Linear Discriminant scores:
        delta_k(x) = x^T P mu_k - 0.5 * mu_k^T P mu_k + log(prior_k)
        where P is precision matrix.
        """
        # X: (n_samples, n_features)
        # means_: (n_classes, n_features)
        # precision_: (n_features, n_features)

        # 1. Compute Weights: W = P @ means_.T -> Shape (n_features, n_classes)
        # However, usually W = means_ @ P is (n_classes, n_features)
        # Let's use W = means_ @ P for calculation: Discriminant = X @ W.T
        W = np.dot(self.means_, self.precision_)  # (n_classes, n_features)

        # 2. Compute Bias: b = -0.5 * diag(means_ @ P @ means_.T) + log(priors)
        # We can reuse W.
        # term1 = diag(means_ @ W.T)
        term1 = np.sum(self.means_ * W, axis=1)  # (n_classes,)
        b = -0.5 * term1 + np.log(
            self.priors_ + 1e-30
        )  # Add epsilon to priors just in case

        # 3. Compute Logits: Z = X @ W.T + b
        logits = np.dot(X, W.T) + b  # (n_samples, n_classes)

        return softmax(logits, axis=1)

    def predict_proba(self, X):
        """
        Predicts class probabilities using float32 arithmetic.

        Args:
            X (np.ndarray): Feature matrix (will be cast to float32).

        Returns:
            np.ndarray: Class probabilities.
        """
        # Ensure input is float32
        X_32 = validate_precision(X, "Inference Input", np.float32)

        # Ensure parameters are float32 (should be done in fit, but safe to check)
        if self.means_.dtype != np.float32:
            self._quantize_parameters()

        return self._predict_proba_internal(X_32)

    def predict(self, X):
        """
        Predicts class labels.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]
