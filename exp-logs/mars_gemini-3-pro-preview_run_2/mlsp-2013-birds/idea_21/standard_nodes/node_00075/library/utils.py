import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Area Under the ROC Curve (AUC) for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth labels (binary matrix).
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: The average ROC AUC score.
    """
    try:
        # Calculate macro-averaged ROC AUC
        return roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Handle edge cases where a specific class might have only one label (all 0s or all 1s)
        # in the provided batch or fold. We calculate AUC only for valid columns.
        valid_columns = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[:, i])) > 1:
                valid_columns.append(i)

        if not valid_columns:
            return 0.5

        return roc_auc_score(
            y_true[:, valid_columns], y_pred[:, valid_columns], average="macro"
        )
