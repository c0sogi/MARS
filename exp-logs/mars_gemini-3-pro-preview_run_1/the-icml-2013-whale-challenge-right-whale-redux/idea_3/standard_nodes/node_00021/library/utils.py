import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import set_seed


def calculate_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (np.array or list): Ground truth binary labels (0 or 1).
        y_pred (np.array or list): Predicted probabilities for the positive class (1).

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    # Ensure inputs are numpy arrays for consistent handling
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for binary classification validity
    # roc_auc_score requires both classes to be present in y_true
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
