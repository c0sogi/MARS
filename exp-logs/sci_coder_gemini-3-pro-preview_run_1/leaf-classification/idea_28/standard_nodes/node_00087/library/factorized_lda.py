import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from library import config, preprocessing


class FactorizedOASLDA:
    """
    Factorized High-Precision OAS Discriminant.

    Decomposes the feature space into semantic groups (Margin, Shape, Texture),
    estimates covariance independently using OAS with geometric consistency,
    and aggregates predictions via Bayesian logit summation.
    """

    def __init__(self):
        self.groups = config.FEATURE_PREFIXES
        self.models = {}
        self.le = LabelEncoder()
        self.priors = None
        self.classes_ = None

    def fit(self, X_dict, y):
        """
        Fits the factorized LDA model.

        Args:
            X_dict (dict): Dictionary of feature matrices (keys: 'margin', 'shape', 'texture').
            y (array-like): Target labels.
        """
        # Encode labels
        y_enc = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        n_classes = len(self.classes_)
        n_samples = len(y)

        # Compute Priors
        counts = np.bincount(y_enc)
        self.priors = counts / n_samples
        # Add epsilon to avoid log(0) for safety, though stratified split prevents this
        self.priors = np.maximum(self.priors, 1e-15)
        log_priors = np.log(self.priors).astype(config.FLOAT_PRECISION)

        print(
            f"Training Factorized OAS LDA on {n_samples} samples with {n_classes} classes..."
        )

        for group in self.groups:
            if group not in X_dict:
                print(f"Warning: Group '{group}' not found in input dictionary.")
                continue

            X_g = X_dict[group]

            # 1. Compute Class Means
            # Shape: (n_classes, n_features)
            means = np.zeros((n_classes, X_g.shape[1]), dtype=config.FLOAT_PRECISION)
            for c in range(n_classes):
                means[c] = X_g[y_enc == c].mean(axis=0)

            # 2. Compute Centered Residuals
            # X_centered = X - mu_y
            X_centered = X_g - means[y_enc]

            # 3. Estimate Precision Matrix via OAS
            # assume_centered=True uses the residuals we computed, ensuring geometric consistency
            oas = OAS(assume_centered=True)
            oas.fit(X_centered)
            precision = oas.precision_.astype(config.FLOAT_PRECISION)

            # 4. Compute Linear Weights and Biases
            # W = Precision @ Means.T  -> Shape (n_features, n_classes)
            W = precision @ means.T

            # Bias term for standard LDA:
            # b_k = -0.5 * (mu_k.T @ Sigma^-1 @ mu_k) + log(pi_k)
            # We compute the quadratic term efficiently:
            # (means @ precision) is (n_classes, n_features)
            # Multiply element-wise with means -> sum over features
            quad_term = -0.5 * np.sum((means @ precision) * means, axis=1)

            # Add log priors to the bias of this expert
            b = quad_term + log_priors

            self.models[group] = (W, b)
            print(f"  Group '{group}': Fitted OAS (Precision shape: {precision.shape})")

        return self

    def predict_proba(self, X_dict):
        """
        Predicts class probabilities using aggregated logits.

        Args:
            X_dict (dict): Dictionary of feature matrices for the test/val set.

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        # Determine n_samples
        first_group = next(iter(self.models))
        n_samples = X_dict[first_group].shape[0]
        n_classes = len(self.classes_)

        # Initialize total logits
        Z_total = np.zeros((n_samples, n_classes), dtype=config.FLOAT_PRECISION)

        # Sum logits from each expert
        for group, (W, b) in self.models.items():
            if group in X_dict:
                X_g = X_dict[group]
                # Linear discriminant: Z = XW + b
                Z_g = X_g @ W + b
                Z_total += Z_g

        # Correction for redundant priors
        # Each Z_g includes log(pi). Summing K experts gives K * log(pi).
        # We want the final logit to contain 1 * log(pi).
        # Therefore, subtract (K - 1) * log(pi).
        log_priors = np.log(self.priors).astype(config.FLOAT_PRECISION)
        n_experts = len(self.models)
        if n_experts > 1:
            correction = (n_experts - 1) * log_priors
            Z_total -= correction

        # Apply Softmax
        probs = softmax(Z_total, axis=1)
        return probs


def run_training_and_submission(debug_size=config.DEBUG_SAMPLE_SIZE):
    """
    Orchestrates the training, validation, and submission process.

    Args:
        debug_size (int or None): Limit dataset size for debugging.
    """
    # 1. Load Data
    print("Loading and preprocessing data...")
    X_train, y_train, X_val, y_val, X_test, ids_test = (
        preprocessing.get_preprocessed_data(
            load_cached_data=True, debug_size=debug_size
        )
    )

    # 2. Train Model
    model = FactorizedOASLDA()
    model.fit(X_train, y_train)

    # 3. Validate
    print("Validating...")
    probs_val = model.predict_proba(X_val)

    # Calculate Log Loss
    # Clip probabilities to avoid log(0) and match competition metric logic
    eps = 1e-15
    probs_val_clipped = np.clip(probs_val, eps, 1 - eps)

    # Get true class indices
    y_val_indices = model.le.transform(y_val)

    # Select probabilities corresponding to true classes
    # Using advanced indexing: probs[row, col]
    true_class_probs = probs_val_clipped[np.arange(len(y_val)), y_val_indices]

    log_loss = -np.mean(np.log(true_class_probs))
    print(f"Validation Log Loss: {log_loss}")

    # 4. Generate Submission
    print("Generating submission...")
    probs_test = model.predict_proba(X_test)

    # Create DataFrame
    submission = pd.DataFrame(probs_test, columns=model.classes_)
    submission.insert(0, config.ID_COLUMN, ids_test)

    # Save
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
