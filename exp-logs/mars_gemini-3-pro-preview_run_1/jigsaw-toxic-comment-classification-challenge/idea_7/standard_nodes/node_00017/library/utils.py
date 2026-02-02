import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true (np.ndarray): Ground truth binary labels with shape (n_samples, n_classes).
        y_pred (np.ndarray): Predicted probabilities with shape (n_samples, n_classes).

    Returns:
        float: The mean column-wise ROC AUC.
    """
    # Calculate ROC AUC for each column and take the average (macro)
    # This corresponds to the "Mean column-wise ROC AUC" metric
    return roc_auc_score(y_true, y_pred, average="macro")
