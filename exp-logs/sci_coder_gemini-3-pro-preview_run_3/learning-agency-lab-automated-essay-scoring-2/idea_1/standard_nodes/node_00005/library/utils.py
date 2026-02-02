import numpy as np
from sklearn.metrics import cohen_kappa_score


def post_process_preds(y_pred):
    """
    Post-processes continuous regression predictions into integer scores.

    Steps:
    1. Clips values to the valid score range [1, 6].
    2. Rounds values to the nearest integer.
    3. Casts values to integers.

    Args:
        y_pred (array-like): Continuous predictions from the model.

    Returns:
        np.ndarray: Integer predictions suitable for submission and metric calculation.
    """
    # Convert to numpy array for efficient element-wise operations
    preds = np.array(y_pred)

    # Clip predictions to the allowed range [1, 6]
    # This handles any outliers produced by the regression model
    preds = np.clip(preds, 1, 6)

    # Round to the nearest integer
    preds = np.round(preds)

    # Return as integers
    return preds.astype(int)


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) metric.

    Args:
        y_true (array-like): Ground truth integer labels.
        y_pred (array-like): Predicted integer labels (output of post_process_preds).

    Returns:
        float: The Quadratic Weighted Kappa score.
    """
    # Ensure inputs are integer numpy arrays
    # This prevents issues if floats are accidentally passed,
    # though y_pred should ideally be post-processed first.
    y_true_int = np.array(y_true, dtype=int)
    y_pred_int = np.array(y_pred, dtype=int)

    # Calculate Cohen's Kappa with quadratic weights
    score = cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")

    return score
