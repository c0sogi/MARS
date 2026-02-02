import numpy as np
from sklearn.metrics import cohen_kappa_score
from library.config import seed_everything


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.

    This function processes the predicted scores by clipping them to the rubric range [1, 6]
    and rounding them to the nearest integer, then calculates the kappa score with quadratic weights.

    Args:
        y_true (array-like): The ground truth scores.
        y_pred (array-like): The predicted scores (can be continuous floats from regression).

    Returns:
        float: The Quadratic Weighted Kappa score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to the valid score range [1, 6] as per the rubric
    y_pred = np.clip(y_pred, 1, 6)

    # Round predictions to the nearest integer to match the ordinal nature of scores
    y_pred = np.round(y_pred)

    # Cast to integers for the metric calculation
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Calculate Quadratic Weighted Kappa
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return qwk
