import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true: Ground truth labels (numpy array or pandas DataFrame).
        y_pred: Predicted probabilities (numpy array or pandas DataFrame).

    Returns:
        float: The mean column-wise ROC AUC.
    """
    # Ensure inputs are numpy arrays
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Calculate ROC AUC for each column and average them
    scores = []
    for i in range(y_true.shape[1]):
        try:
            col_score = roc_auc_score(y_true[:, i], y_pred[:, i])
            scores.append(col_score)
        except ValueError:
            # Handle edge cases where a column has only one class
            continue

    if len(scores) == 0:
        return 0.0

    return np.mean(scores)
