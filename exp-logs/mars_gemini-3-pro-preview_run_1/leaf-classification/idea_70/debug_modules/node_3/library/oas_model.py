import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import OAS
from sklearn.metrics import log_loss, accuracy_score
from library.config import NP_FLOAT_PRECISION, FLOAT_PRECISION, DEVICE, SEED


class OASDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Classifier using Oracle Approximating Shrinkage (OAS)
    for covariance estimation.

    Implements a hybrid CPU-training / GPU-inference pipeline for maximum
    numerical precision (float64).
    """

    def __init__(self):
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.precision_ = None
        self.W_ = None  # Weights: (n_classes, n_features)
        self.b_ = None  # Bias: (n_classes,)

    def fit(self, X, y):
        """
        Fits the OAS Discriminant model.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features).
            y (np.ndarray): Training labels, shape (n_samples,).

        Returns:
            self
        """
        # Ensure high precision inputs
        X = X.astype(NP_FLOAT_PRECISION)

        # 1. Parameter Estimation
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        self.means_ = np.zeros((n_classes, n_features), dtype=NP_FLOAT_PRECISION)
        self.priors_ = np.zeros(n_classes, dtype=NP_FLOAT_PRECISION)

        # Compute means and priors
        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.means_[idx] = np.mean(X_cls, axis=0)
            self.priors_[idx] = len(X_cls) / n_samples

        # 2. Covariance Estimation via OAS
        # Compute residuals (centering data by class means)
        # Map y to indices 0..K-1 for fast indexing
        # We assume y is already label encoded 0..K-1 or we map it.
        # However, to be safe with arbitrary labels, we map using classes_
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        y_indices = np.array([class_to_idx[yi] for yi in y])

        residuals = X - self.means_[y_indices]

        # Fit OAS on residuals
        # assume_centered=True because residuals have zero mean by definition
        oas = OAS(assume_centered=True)
        oas.fit(residuals)

        self.precision_ = oas.precision_.astype(NP_FLOAT_PRECISION)

        # 3. Weight Derivation (Linearization)
        # Discriminant function: d_k(x) = x.T @ (P @ mu_k) + (log(pi_k) - 0.5 * mu_k.T @ P @ mu_k)
        # W_k = P @ mu_k
        # b_k = log(pi_k) - 0.5 * mu_k.T @ W_k

        # W shape: (n_classes, n_features)
        # self.means_ shape: (n_classes, n_features)
        # self.precision_ shape: (n_features, n_features)

        # Transpose logic: (P @ mu)^T = mu^T @ P^T = mu^T @ P (since P is symmetric)
        self.W_ = np.dot(self.means_, self.precision_)

        # Compute quadratic term: diag(mu @ P @ mu.T) -> diag(mu @ W.T)
        # We only need the diagonal elements: sum(mu * W, axis=1)
        quadratic_term = np.sum(self.means_ * self.W_, axis=1)

        self.b_ = np.log(self.priors_) - 0.5 * quadratic_term

        return self

    def get_linear_parameters(self):
        """
        Returns the derived linear parameters W and b.
        """
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.W_, self.b_, self.classes_

    def predict_proba(self, X):
        """
        Predicts class probabilities using GPU-accelerated float64 inference.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities.
        """
        if self.W_ is None:
            raise RuntimeError("Model is not fitted yet.")

        # Ensure input is float64
        X = X.astype(NP_FLOAT_PRECISION)

        # Move data and weights to GPU (PyTorch)
        # Using no_grad for inference
        with torch.no_grad():
            # Convert to tensors
            # Note: library.config.FLOAT_PRECISION is torch.float64
            X_t = torch.tensor(X, dtype=FLOAT_PRECISION, device=DEVICE)
            W_t = torch.tensor(self.W_, dtype=FLOAT_PRECISION, device=DEVICE)
            b_t = torch.tensor(self.b_, dtype=FLOAT_PRECISION, device=DEVICE)

            # Linear Scoring: Z = X @ W.T + b
            # X: (N, D), W: (K, D), b: (K,)
            # Result: (N, K)
            logits = F.linear(X_t, W_t, b_t)

            # Softmax
            probs_t = F.softmax(logits, dim=1)

            # Move back to CPU
            probs = probs_t.cpu().numpy()

        return probs

    def predict(self, X):
        """
        Predicts class labels.
        """
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


def evaluate_log_loss(y_true, y_prob):
    """
    Calculates multi-class log loss with specific clipping as per task description.
    Range: [1e-15, 1 - 1e-15]
    """
    # Rescale rows to sum to 1 (handled by softmax, but good for safety)
    row_sums = y_prob.sum(axis=1)
    y_prob = y_prob / row_sums[:, np.newaxis]

    # Clip
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    return log_loss(y_true, y_prob)


def train_oas_model(data_dict, verbose=True):
    """
    Orchestrates the training and evaluation of the OAS Discriminant.

    Args:
        data_dict (dict): Dictionary containing processed data from data_pipeline.
        verbose (bool): Whether to print metrics.

    Returns:
        OASDiscriminant: The fitted model.
    """
    X_train, y_train = data_dict["train"]
    X_val, y_val = data_dict["val"]

    if verbose:
        print(
            f"Training OAS Discriminant on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
        )
        print(f"Device: {DEVICE}, Precision: {FLOAT_PRECISION}")

    model = OASDiscriminant()
    model.fit(X_train, y_train)

    # Evaluation
    if verbose:
        print("Evaluating model...")

    # Train Metrics
    train_probs = model.predict_proba(X_train)
    train_loss = evaluate_log_loss(y_train, train_probs)
    train_acc = accuracy_score(y_train, model.predict(X_train))

    # Val Metrics
    val_probs = model.predict_proba(X_val)
    val_loss = evaluate_log_loss(y_val, val_probs)
    val_acc = accuracy_score(y_val, model.predict(X_val))

    if verbose:
        print("-" * 40)
        print(f"Train Log Loss: {train_loss:.16f}")
        print(f"Train Accuracy: {train_acc:.16f}")
        print(f"Val Log Loss:   {val_loss:.16f}")
        print(f"Val Accuracy:   {val_acc:.16f}")
        print("-" * 40)

    return model
