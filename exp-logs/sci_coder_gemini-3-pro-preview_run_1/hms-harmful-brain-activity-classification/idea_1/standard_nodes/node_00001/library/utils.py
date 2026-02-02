import numpy as np
import pandas as pd
import os
from library.config import set_seed, setup_directories


def compute_kl_divergence(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15
) -> float:
    """
    Computes the Kullback-Leibler divergence between the ground truth and predicted probabilities.

    The metric is defined as: KL(P || Q) = sum(P * log(P / Q))
    where P is the ground truth distribution and Q is the predicted distribution.

    Args:
        y_true: Ground truth probabilities (P), shape (N, C).
        y_pred: Predicted probabilities (Q), shape (N, C).
        epsilon: Small constant to avoid numerical instability (log(0)).

    Returns:
        float: The average KL divergence across all samples.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Normalize predicted rows to sum to 1 (safety check)
    # This ensures Q is a valid probability distribution
    pred_sums = np.sum(y_pred, axis=1, keepdims=True)
    pred_sums[pred_sums == 0] = 1.0
    y_pred = y_pred / pred_sums

    # Compute KL Divergence terms
    # Term 1: sum(P * log(P))
    # We handle the case where P=0 by defining 0*log(0) = 0
    term1 = np.zeros_like(y_true)
    mask = y_true > 0
    term1[mask] = y_true[mask] * np.log(y_true[mask])

    # Term 2: sum(P * log(Q))
    term2 = y_true * np.log(y_pred)

    # KL = sum(P * log(P) - P * log(Q))
    # Sum over classes (axis 1)
    kl_per_sample = np.sum(term1 - term2, axis=1)

    # Return mean over samples
    return np.mean(kl_per_sample)


def normalize_probabilities(preds: np.ndarray) -> np.ndarray:
    """
    Ensures that the predicted probabilities sum to exactly 1.0 for each row.
    This is critical for meeting the submission format requirements.

    Args:
        preds: Raw predicted probabilities or logits, shape (N, C).

    Returns:
        np.ndarray: Normalized probabilities.
    """
    preds = np.asarray(preds, dtype=np.float64)

    # Ensure non-negative
    preds = np.maximum(preds, 0)

    row_sums = np.sum(preds, axis=1, keepdims=True)

    # Handle rows that sum to 0 (assign uniform probability)
    zero_sum_mask = (row_sums == 0).flatten()
    if np.any(zero_sum_mask):
        n_classes = preds.shape[1]
        preds[zero_sum_mask] = 1.0 / n_classes
        row_sums[zero_sum_mask] = 1.0

    return preds / row_sums


def verify_submission_format(df: pd.DataFrame, required_cols: list) -> bool:
    """
    Verifies that the submission DataFrame complies with the competition format.

    Args:
        df: The submission DataFrame.
        required_cols: List of column names that must be present (vote columns).

    Returns:
        bool: True if format is valid, False otherwise.
    """
    # 1. Check for required columns
    if not all(col in df.columns for col in required_cols):
        print(f"Validation Error: Missing columns. Expected {required_cols}")
        return False

    # 2. Check for NaN values
    if df[required_cols].isnull().any().any():
        print("Validation Error: Submission contains NaN values.")
        return False

    # 3. Check if probabilities sum to 1.0
    probs = df[required_cols].values
    row_sums = np.sum(probs, axis=1)
    # Allow small floating point error
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        print("Validation Error: Rows do not sum to 1.0.")
        return False

    return True
