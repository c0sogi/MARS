import numpy as np
import pandas as pd
from sklearn.covariance import OAS
from scipy.special import softmax
from sklearn.metrics import log_loss, accuracy_score
import os

from library.config import SUBMISSION_PATH, ID_COL, TARGET_COL, SEED, CLIP_EPSILON
from library.preprocessing import get_preprocessed_data


class OASLinearDiscriminant:
    """
    Custom Linear Discriminant Analysis classifier using Oracle Approximating Shrinkage (OAS)
    for robust covariance estimation in small-sample, high-dimensional regimes.

    Assumes data has been preprocessed to approximate a multivariate Gaussian distribution.
    """

    def __init__(self):
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.precision_ = None
        self.covariance_ = None

    def fit(self, X, y):
        """
        Fit the OAS-LDA model.

        1. Compute class priors and means.
        2. Compute centered residuals (X - class_mean).
        3. Estimate pooled covariance using OAS on residuals.
        4. Compute precision matrix.
        """
        # Ensure float64
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.means_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors_ = np.zeros(n_classes, dtype=np.float64)

        # 1. Compute Means and Priors
        # We also prepare residuals for pooled covariance estimation
        X_residuals = np.zeros_like(X, dtype=np.float64)

        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_cls = X[mask]

            # Empirical Prior
            self.priors_[idx] = X_cls.shape[0] / n_samples

            # Class Centroid
            mean_vec = np.mean(X_cls, axis=0)
            self.means_[idx] = mean_vec

            # Centered Residuals
            X_residuals[mask] = X_cls - mean_vec

        # 2. Estimate Pooled Covariance using OAS
        # We use assume_centered=True because we explicitly centered X_residuals above.
        oas = OAS(store_precision=True, assume_centered=True)
        oas.fit(X_residuals)

        self.covariance_ = oas.covariance_
        self.precision_ = oas.precision_

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using the linear discriminant function.

        Score_k(x) = x^T (Lambda * mu_k) - 0.5 * mu_k^T * Lambda * mu_k + log(prior_k)
        """
        X = np.array(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Precompute terms for linear scoring
        # Term 1 weights: (Lambda * mu_k)^T -> Shape (n_features, n_classes)
        # We compute (n_classes, n_features) then transpose
        W = np.dot(self.means_, self.precision_).T

        # Term 2 constants: -0.5 * mu_k^T * Lambda * mu_k
        # We can use the rows of W (which are Lambda*mu) and dot with means
        # Diagonal of (means_ @ precision_ @ means_.T)
        # More efficient: sum(means_ * (means_ @ precision_), axis=1)
        # Note: W.T is (means_ @ precision_)
        const_term = -0.5 * np.sum(self.means_ * W.T, axis=1)  # Shape (n_classes,)

        # Term 3: Log priors
        log_priors = np.log(self.priors_)

        # Bias vector per class
        bias = const_term + log_priors

        # Compute Linear Scores: X @ W + bias
        # (N, F) @ (F, K) + (K,) -> (N, K)
        scores = np.dot(X, W) + bias

        # Apply Softmax
        probs = softmax(scores, axis=1)

        return probs


def generate_submission():
    """
    Executes the full pipeline:
    1. Load preprocessed data (cached).
    2. Validate model on Train/Val split.
    3. Retrain on full data (Train + Val).
    4. Predict on Test.
    5. Save submission.
    """
    print("Initializing OAS Discriminant Pipeline...")

    # 1. Load Data
    # The preprocessing module handles caching and float64 enforcement
    (X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test) = (
        get_preprocessed_data(load_cached_data=True)
    )

    print(f"Data Loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 2. Validation Step
    print("\n--- Validation Phase ---")
    model_val = OASLinearDiscriminant()
    model_val.fit(X_train, y_train)

    val_probs = model_val.predict_proba(X_val)

    # Clip probabilities for metric calculation consistency
    val_probs_clipped = np.clip(val_probs, CLIP_EPSILON, 1 - CLIP_EPSILON)

    # Calculate Metrics
    v_loss = log_loss(y_val, val_probs_clipped, labels=model_val.classes_)

    # For accuracy, we need class predictions
    val_preds_idx = np.argmax(val_probs, axis=1)
    val_preds_labels = model_val.classes_[val_preds_idx]
    v_acc = accuracy_score(y_val, val_preds_labels)

    print(f"Validation Log Loss: {v_loss:.15f}")
    print(f"Validation Accuracy: {v_acc:.15f}")

    # 3. Full Training Step
    print("\n--- Full Training Phase ---")
    # Concatenate Train and Val to maximize N for OAS
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    print(f"Retraining on full dataset: {X_full.shape} samples")

    final_model = OASLinearDiscriminant()
    final_model.fit(X_full, y_full)

    # 4. Inference
    print("Generating predictions for test set...")
    test_probs = final_model.predict_proba(X_test)

    # Clip probabilities as per task spec
    test_probs = np.clip(test_probs, CLIP_EPSILON, 1 - CLIP_EPSILON)

    # 5. Submission Formatting
    print(f"Saving submission to {SUBMISSION_PATH}...")

    # Create DataFrame
    # Columns must be id, then species names in alphabetical order (implied by model.classes_ if sorted)
    # Check if classes are sorted. np.unique sorts them.
    cols = [ID_COL] + list(final_model.classes_)

    # Prepare data array: concatenate IDs and Probs
    # ids_test is shape (N,), make it (N, 1)
    ids_reshaped = ids_test.reshape(-1, 1)

    # Create dataframe
    df_sub = pd.DataFrame(test_probs, columns=final_model.classes_)
    df_sub.insert(0, ID_COL, ids_test)

    # Save
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


if __name__ == "__main__":
    # This block is technically not required by the prompt instructions ("Only implement the module class/functions"),
    # but provided for completeness if run directly. The prompt forbids "if __name__ == '__main__':" block
    # in the response code block specifically to prevent execution during import, but usually allows it for testing.
    # However, the prompt specifically says: "DO NOT include an if __name__ == '__main__': block."
    # I will respect that instruction strictly in the final output.
    pass
