import numpy as np
from sklearn.metrics import log_loss
from library.config import PROB_CLIP


def calculate_log_loss(y_true, y_pred, labels=None):
    """
    Calculates the multi-class log loss metric with specific rescaling and clipping
    as defined in the task description.

    The process involves:
    1. Rescaling: Each row of probabilities is divided by its sum.
    2. Clipping: Probabilities are clipped to [1e-15, 1 - 1e-15].
    3. Scoring: Multi-class log loss is computed.

    Args:
        y_true (array-like): Ground truth labels (n_samples,). Can be class indices or names.
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).
        labels (array-like, optional): List of class labels to index the matrix.
                                       Required if y_true are strings or integers not
                                       spanning the full range 0..n_classes-1.

    Returns:
        float: The calculated log loss.
    """
    # Ensure y_pred is a float64 numpy array for precision
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale: Divide each row by the row sum
    # This ensures probabilities sum to 1, as per competition scoring rules.
    row_sums = y_pred.sum(axis=1, keepdims=True)
    # Handle potential zero sums to avoid NaN (though unlikely with proper models)
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums

    # 2. Clip: Restrict probabilities to avoid log(0)
    # Range: [PROB_CLIP, 1 - PROB_CLIP]
    y_pred = np.clip(y_pred, PROB_CLIP, 1 - PROB_CLIP)

    # 3. Calculate Log Loss
    # We pass the processed probabilities to sklearn's log_loss.
    # Note: sklearn also has an internal 'eps' parameter (default 1e-15),
    # but we perform explicit clipping to strictly satisfy the task description.
    score = log_loss(y_true, y_pred, labels=labels)

    return score
