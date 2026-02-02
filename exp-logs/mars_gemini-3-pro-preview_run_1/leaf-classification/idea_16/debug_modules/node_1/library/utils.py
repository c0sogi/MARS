import os
import random
import numpy as np
from sklearn.metrics import log_loss
from library.config import PROB_CLIP_MIN, PROB_CLIP_MAX


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and environment variables.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def normalize_and_clip_probabilities(y_pred: np.ndarray) -> np.ndarray:
    """
    Rescales probabilities to sum to 1 per row and clips them to avoid log(0),
    matching the competition metric definition.

    Args:
        y_pred (np.ndarray): Raw predicted probabilities.

    Returns:
        np.ndarray: Processed probabilities ready for scoring.
    """
    # Ensure input is a numpy array of floats
    preds = np.array(y_pred, dtype=np.float64)

    # Rescale rows to sum to 1
    row_sums = preds.sum(axis=1, keepdims=True)

    # Handle rows with zero sum by assigning uniform probability
    # This ensures that even empty predictions result in a valid probability distribution
    zero_sum_mask = (row_sums == 0).flatten()
    if np.any(zero_sum_mask):
        preds[zero_sum_mask] = 1.0 / preds.shape[1]
        row_sums[zero_sum_mask] = 1.0

    preds = preds / row_sums

    # Clip probabilities to the specified range [1e-15, 1 - 1e-15]
    preds = np.clip(preds, PROB_CLIP_MIN, PROB_CLIP_MAX)

    return preds


def compute_log_loss(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list = None
) -> float:
    """
    Calculates the multi-class log loss after applying the specific normalization
    and clipping rules of the task.

    Args:
        y_true (np.ndarray): True class labels (indices or one-hot).
        y_pred (np.ndarray): Predicted probabilities.
        labels (list, optional): List of class labels if y_true are strings.

    Returns:
        float: The calculated log loss.
    """
    # Apply metric-specific preprocessing
    preds_processed = normalize_and_clip_probabilities(y_pred)

    # Calculate Log Loss
    # Note: Scikit-learn's log_loss internally clips as well, but we do it explicitly
    # to match the specific competition rules (rescaling + specific epsilon).
    score = log_loss(y_true, preds_processed, labels=labels)
    return score
