import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import seed_everything, Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the centralized seed_everything function in library.config.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    seed_everything(seed)


def compute_metrics(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC) for binary classification.

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities for the positive class (1).

    Returns:
        float: The Area Under the ROC Curve.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate AUC
    # Note: roc_auc_score raises a ValueError if y_true contains only one class.
    # We allow this to propagate so the training loop is aware of invalid splits.
    auc = roc_auc_score(y_true, y_pred)

    return auc
