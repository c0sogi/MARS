import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import SEED


def set_seed(seed: int = SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_pred) -> float:
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted probabilities for the positive class.

    Returns:
        float: The computed AUC score.
    """
    # Check if y_true contains only one class to avoid ValueError from sklearn
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)
