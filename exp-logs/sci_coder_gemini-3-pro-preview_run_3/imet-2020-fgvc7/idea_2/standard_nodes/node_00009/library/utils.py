import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_f1(y_true, y_pred):
    """
    Calculates the Micro-F1 score.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred (np.array): Predicted binary labels.

    Returns:
        float: The Micro-F1 score.
    """
    return f1_score(y_true, y_pred, average="micro")


def optimize_threshold(y_true, y_pred_probs):
    """
    Finds the optimal probability threshold that maximizes the Micro-F1 score.

    Args:
        y_true (np.array): Ground truth binary labels.
        y_pred_probs (np.array): Predicted probabilities (output of sigmoid).

    Returns:
        tuple: (best_threshold, best_f1_score)
    """
    best_threshold = 0.5
    best_f1 = 0.0

    # Iterate through thresholds from 0.01 to 0.99
    thresholds = np.arange(0.01, 1.00, 0.01)

    for thresh in thresholds:
        # Binarize predictions based on current threshold
        y_pred_bin = (y_pred_probs > thresh).astype(int)

        # Calculate score
        score = calculate_f1(y_true, y_pred_bin)

        if score > best_f1:
            best_f1 = score
            best_threshold = thresh

    print(
        f"Threshold Optimization Completed. Best Threshold: {best_threshold}, Best F1: {best_f1}"
    )
    return best_threshold, best_f1
