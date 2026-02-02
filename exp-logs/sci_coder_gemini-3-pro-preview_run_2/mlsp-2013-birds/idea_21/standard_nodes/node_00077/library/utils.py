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
    Manually iterates over classes to handle sparse labels robustly.

    Args:
        y_true (np.ndarray): Ground truth labels (binary matrix).
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: The average ROC AUC score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    scores = []
    # Iterate over each class
    for i in range(y_true.shape[1]):
        # Only calculate AUC if the class has both 0s and 1s in this batch/fold
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                scores.append(score)
            except ValueError:
                pass

    if not scores:
        return 0.5

    return np.mean(scores)
