import os
import random
import numpy as np
import torch


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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Where:
        pTP = sum(y_pred * y_true)
        pFP = sum(y_pred * (1 - y_true))
        TP + FN = sum(y_true)

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities in range [0, 1].
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The probabilistic F1 score.
    """
    # Convert to numpy arrays if they are lists or torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Probabilistic True Positives (pTP)
    p_tp = np.sum(y_pred * y_true)

    # Probabilistic False Positives (pFP)
    p_fp = np.sum(y_pred * (1 - y_true))

    # Total Positives (TP + FN) - actual positive count
    total_positives = np.sum(y_true)

    # Probabilistic Precision
    # pPrecision = pTP / (pTP + pFP) = pTP / sum(y_pred)
    # Note: pTP + pFP = sum(y_pred * y_true + y_pred - y_pred * y_true) = sum(y_pred)
    denominator_precision = p_tp + p_fp
    p_precision = p_tp / (denominator_precision + epsilon)

    # Probabilistic Recall
    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # Probabilistic F1
    denominator_f1 = p_precision + p_recall
    pf1 = 2 * (p_precision * p_recall) / (denominator_f1 + epsilon)

    return pf1
