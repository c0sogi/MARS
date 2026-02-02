import numpy as np
from sklearn.metrics import cohen_kappa_score


def compute_qwk(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa (QWK) score.

    This function computes the quadratic weighted kappa, which measures the
    agreement between two ratings. This metric typically varies from 0 (random agreement)
    to 1 (complete agreement). In the event that there is less agreement than expected
    by chance, the metric may be negative.

    Args:
        y_true (array-like): The ground truth labels (integers).
        y_pred (array-like): The predicted labels (integers).
                             Floats will be cast to integers.

    Returns:
        float: The quadratic weighted kappa score.
    """
    # Ensure inputs are numpy arrays and cast to integer type
    # This handles cases where predictions might be passed as floats or lists
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Calculate QWK using sklearn's implementation with quadratic weights
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return qwk
