import numpy as np
from sklearn.metrics import log_loss
from library.config import set_deterministic


def seed_everything(seed: int):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    This function wraps the centralized deterministic setting from the config library.

    Args:
        seed (int): The seed value to use.
    """
    set_deterministic(seed)


def calculate_log_loss(y_true, y_pred):
    """
    Computes the Log Loss metric for binary classification.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate log loss.
    # labels=[0, 1] ensures the metric is calculated correctly even if
    # the specific batch/subset is missing one of the classes.
    return log_loss(y_true, y_pred, labels=[0, 1])
