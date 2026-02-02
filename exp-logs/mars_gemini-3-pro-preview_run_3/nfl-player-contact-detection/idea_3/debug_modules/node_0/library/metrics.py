import numpy as np
from library.utils import get_logger, compute_mcc

logger = get_logger("metrics")


def calculate_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.

    Returns:
        float: The MCC score.
    """
    return compute_mcc(y_true, y_pred)


def optimize_threshold(y_true, y_pred_proba, step=0.01):
    """
    Finds the best probability threshold to maximize MCC using a linear search.

    This function iterates through probability thresholds from `step` to 1.0,
    converts probabilities to binary predictions, and calculates the MCC.
    It returns the threshold that yields the highest MCC score.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred_proba (array-like): Predicted probabilities for the positive class.
        step (float): Step size for the threshold search. Defaults to 0.01.

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    # Ensure inputs are numpy arrays for efficient broadcasting/indexing
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    logger.info("Optimizing decision threshold based on MCC...")

    # Generate thresholds from step up to 1.0
    thresholds = np.arange(step, 1.0, step)

    best_mcc = -1.0
    best_thresh = 0.5

    # Iterate over all thresholds to find the optimum
    for thresh in thresholds:
        # Apply threshold to generate binary predictions
        y_pred_binary = (y_pred_proba >= thresh).astype(int)

        # Calculate metric using the imported utility
        mcc = calculate_mcc(y_true, y_pred_binary)

        # Update best score
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = thresh

    logger.info(f"Optimal Threshold: {best_thresh}")
    logger.info(f"Best Validation MCC: {best_mcc}")

    return best_thresh, best_mcc
