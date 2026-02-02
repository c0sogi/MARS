import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def probabilistic_f1(y_true, y_pred):
    """
    Calculates the Probabilistic F1 score (pF1) as defined in the task.

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)

    Where:
    pPrecision = pTP / (pTP + pFP)
    pRecall = pTP / (TP + FN)

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred: Array-like of predicted probabilities (0.0 to 1.0).

    Returns:
        float: The calculated pF1 score.
    """
    # Convert inputs to numpy arrays for vectorized operations
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}."
        )

    # Calculate Probabilistic True Positives (pTP)
    # Sum of probabilities for actual positive cases
    p_tp = np.sum(y_true * y_pred)

    # Calculate Probabilistic False Positives (pFP)
    # Sum of probabilities for actual negative cases
    p_fp = np.sum((1 - y_true) * y_pred)

    # Calculate Total Actual Positives (TP + FN)
    # This is simply the count of positive samples in the ground truth
    total_actual_positives = np.sum(y_true)

    # Calculate Probabilistic Precision
    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP is mathematically equivalent to sum(y_pred)
    denominator_precision = p_tp + p_fp

    if denominator_precision == 0:
        p_precision = 0.0
    else:
        p_precision = p_tp / denominator_precision

    # Calculate Probabilistic Recall
    # pRecall = pTP / (TP + FN)
    if total_actual_positives == 0:
        p_recall = 0.0
    else:
        p_recall = p_tp / total_actual_positives

    # Calculate Probabilistic F1
    # pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    denominator_f1 = p_precision + p_recall

    if denominator_f1 == 0:
        p_f1 = 0.0
    else:
        p_f1 = 2 * (p_precision * p_recall) / denominator_f1

    return p_f1
