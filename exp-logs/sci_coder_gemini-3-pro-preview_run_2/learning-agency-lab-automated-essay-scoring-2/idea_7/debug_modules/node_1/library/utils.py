import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays/lists and integer types if necessary,
    # though cohen_kappa_score handles various inputs.
    # The metric requires discrete integer labels for the confusion matrix calculation
    # implicit in kappa, but we pass weights='quadratic'.
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")
