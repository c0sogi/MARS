import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_auc(y_true, y_pred):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores, can either be probability estimates of the positive class,
                             confidence values, or non-thresholded measure of decisions.

    Returns:
        float: The ROC AUC score.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        raise ValueError("y_true is empty.")

    return roc_auc_score(y_true, y_pred)
