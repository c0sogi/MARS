import numpy as np
from sklearn.covariance import OAS
from library.config import NUMERIC_DTYPE, FEATURE_PREFIXES
from library.utils import calculate_log_loss, save_submission
from library.data_loader import load_and_process_data


class FactorizedOASDiscriminant:
    """
    A Bayesian Ensemble of Linear Discriminants that factorizes the feature space
    into semantic groups (Margin, Shape, Texture).

    Each group is modeled by an independent LDA expert using OAS covariance estimation.
    Predictions are aggregated by summing logits (product of likelihoods) and
    correcting for prior over-counting.
    """

    def __init__(self):
        self.experts = {}
        self.log_priors = None
        self.classes = None

    def fit(self, X_dict, y):
        """
        Fits the factorized model.

        Args:
            X_dict (dict): Dictionary containing feature arrays for each group.
            y (array): Target labels (integers).
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # 1. Calculate Global Priors
        # Assumes y contains integers 0 to K-1
        counts = np.bincount(y)
        # Add epsilon to avoid log(0) if a class is missing (unlikely in valid split)
        priors = counts / counts.sum()
        self.log_priors = np.log(priors + 1e-15).astype(NUMERIC_DTYPE)

        # 2. Train Independent Experts per Feature Group
        for group in FEATURE_PREFIXES:
            # Ensure high precision
            X_g = X_dict[group].astype(NUMERIC_DTYPE)
            n_samples, n_features = X_g.shape

            # A. Compute Class Means
            means = np.zeros((n_classes, n_features), dtype=NUMERIC_DTYPE)
            for k in self.classes:
                means[k] = np.mean(X_g[y == k], axis=0)

            # B. Center Data (Compute Residuals)
            # OAS assumes centered data if assume_centered=True
            X_centered = np.zeros_like(X_g)
            for k in self.classes:
                X_centered[y == k] = X_g[y == k] - means[k]

            # C. Estimate Precision Matrix via OAS
            # We use the residuals to estimate the common within-class covariance
            oas = OAS(assume_centered=True)
            oas.fit(X_centered)
            precision = oas.precision_.astype(NUMERIC_DTYPE)

            # D. Compute Linear Discriminant Parameters
            # Standard LDA Discriminant: delta_k(x) = x.T @ W.T + b
            # Where W = means @ precision
            W = np.dot(means, precision)  # Shape: (n_classes, n_features)

            # Quadratic Bias term: -0.5 * diag(means @ precision @ means.T)
            # Efficiently computed as row-wise dot product of means and W
            quad_term = -0.5 * np.sum(means * W, axis=1)

            # Expert Bias includes the log_prior (standard LDA formulation)
            b = quad_term + self.log_priors

            self.experts[group] = {"W": W, "b": b}

    def predict_proba(self, X_dict):
        """
        Predicts class probabilities using Bayesian Aggregation.
        """
        # Determine shapes
        first_group = list(X_dict.keys())[0]
        n_samples = X_dict[first_group].shape[0]
        n_classes = len(self.classes)

        # Initialize total logits
        total_logits = np.zeros((n_samples, n_classes), dtype=NUMERIC_DTYPE)

        # Sum logits from all experts
        for group in FEATURE_PREFIXES:
            X_g = X_dict[group].astype(NUMERIC_DTYPE)
            expert = self.experts[group]

            W = expert["W"]  # (K, F)
            b = expert["b"]  # (K,)

            # Compute expert logits: X @ W.T + b
            # (N, F) @ (F, K) -> (N, K)
            logits = np.dot(X_g, W.T) + b
            total_logits += logits

        # Bayesian Correction
        # Since each expert adds log(prior), we have summed it N times.
        # We need it exactly once. Subtract (N-1) * log(prior).
        n_experts = len(FEATURE_PREFIXES)
        correction = (n_experts - 1) * self.log_priors
        final_logits = total_logits - correction

        # Softmax in high precision
        # Shift logits for numerical stability
        final_logits_shifted = final_logits - np.max(
            final_logits, axis=1, keepdims=True
        )
        exp_logits = np.exp(final_logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs


def run_factorized_lda_pipeline():
    """
    Orchestrates the data loading, training, validation, and submission generation.
    """
    print("Initializing Factorized OAS Discriminant Pipeline...")

    # 1. Load and Process Data
    # This handles semantic splitting and caching automatically
    data, class_names = load_and_process_data(load_cached_data=True)

    # 2. Train Model
    print("Training Factorized Experts...")
    model = FactorizedOASDiscriminant()
    model.fit(data["train"]["X"], data["train"]["y"])

    # 3. Validate
    print("Evaluating on Validation Set...")
    val_probs = model.predict_proba(data["val"]["X"])
    val_loss = calculate_log_loss(data["val"]["y"], val_probs)
    print(f"Validation Multi-class Log Loss: {val_loss}")

    # 4. Generate Submission
    print("Generating Test Predictions...")
    test_probs = model.predict_proba(data["test"]["X"])
    save_submission(data["test"]["ids"], test_probs, class_names)
    print("Pipeline Completed Successfully.")
