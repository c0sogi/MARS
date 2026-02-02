import os
import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from library import config, preprocessing


class GlobalOASLDA:
    """
    Global High-Precision OAS Discriminant.
    Cite solution_lesson_node_00087: Prefer global shrinkage estimators.
    """

    def __init__(self):
        self.le = LabelEncoder()
        self.priors = None
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Encode labels
        y_enc = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        print(
            f"Training Global OAS LDA on {n_samples} samples with {n_classes} classes..."
        )

        # Compute Priors
        # Cite solution_lesson_node_00033: Use empirical priors
        self.priors = np.bincount(y_enc) / n_samples

        # 1. Compute Class Means (Arithmetic Mean)
        # Cite solution_lesson_node_00066: Optimality of Arithmetic Mean
        means = np.zeros((n_classes, n_features), dtype=config.FLOAT_PRECISION)
        for c in range(n_classes):
            means[c] = X[y_enc == c].mean(axis=0)

        # 2. Compute Centered Residuals
        X_centered = X - means[y_enc]

        # 3. Estimate Precision Matrix via OAS
        # Cite solution_lesson_node_00061: assume_centered=True
        # Cite solution_lesson_node_00047: Use OAS over Ledoit-Wolf
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)
        # Cite solution_lesson_node_00062: Use precision_ attribute
        precision = oas.precision_.astype(config.FLOAT_PRECISION)

        # 4. Compute Linear Weights (Linear Formulation)
        # Cite solution_lesson_node_00055: Prefer Linear Formulation
        self.coef_ = precision @ means.T  # Shape (n_features, n_classes)

        # 5. Compute Bias
        # b_k = -0.5 * (mu_k.T @ Sigma^-1 @ mu_k) + log(pi_k)
        quad_term = -0.5 * np.sum((means @ precision) * means, axis=1)
        self.intercept_ = quad_term + np.log(self.priors)

        return self

    def predict_proba(self, X):
        # Z = X @ W + b
        # Cite solution_lesson_node_00057: Inference precision dictates performance ceiling
        scores = X @ self.coef_ + self.intercept_
        # Cite solution_lesson_node_00085: Use optimized library primitives
        return softmax(scores, axis=1)


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
