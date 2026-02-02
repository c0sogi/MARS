import numpy as np
import torch
from library.config import set_seed


def seed_everything(seed):
    """
    Sets the random seed for reproducibility using the provided library configuration.

    Args:
        seed (int): The seed value to set.
    """
    set_seed(seed)


def probabilistic_f1(y_true, y_pred, beta=1.0, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1) as defined in the task description.

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    where:
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)
        pTP = sum(y_true * y_pred)
        pFP = sum((1 - y_true) * y_pred)
        TP + FN = sum(y_true)

    Args:
        y_true (array-like or torch.Tensor): Ground truth labels (0 or 1).
        y_pred (array-like or torch.Tensor): Predicted probabilities (0.0 to 1.0).
        beta (float): The beta parameter for the F-score (default 1.0).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The probabilistic F1 score.
    """
    # Handle PyTorch tensors by detaching and moving to CPU
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Input validation
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}"
        )

    # Calculate Probabilistic True Positives (pTP)
    # Sum of probabilities for the positive class where ground truth is positive
    p_tp = np.sum(y_true * y_pred)

    # Calculate Probabilistic False Positives (pFP)
    # Sum of probabilities for the positive class where ground truth is negative
    p_fp = np.sum((1 - y_true) * y_pred)

    # Calculate Total Positives (TP + FN)
    # The actual count of positive cases in the ground truth
    total_positives = np.sum(y_true)

    # Calculate Probabilistic Precision
    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP is mathematically equivalent to sum(y_pred)
    denominator_precision = p_tp + p_fp
    p_precision = p_tp / (denominator_precision + epsilon)

    # Calculate Probabilistic Recall
    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # Calculate Probabilistic F-beta Score
    # Formula: (1 + beta^2) * (P * R) / ((beta^2 * P) + R)
    beta_sq = beta**2
    numerator = (1 + beta_sq) * (p_precision * p_recall)
    denominator = (beta_sq * p_precision) + p_recall

    pf1 = numerator / (denominator + epsilon)

    return float(pf1)
