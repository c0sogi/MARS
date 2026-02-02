import os
import sys
import torch
import numpy as np

# Ensure the library module can be found
sys.path.append(os.getcwd())
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the configuration library's set_seed function to avoid re-implementation.

    Args:
        seed (int): The seed value to use.
    """
    set_seed(seed)


def get_device():
    """
    Determines the available computational device.

    Returns:
        torch.device: Returns 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    The metric is an extension of the traditional F score that accepts probabilities
    instead of binary classifications.

    Formulas:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Where:
        pTP = Sum(y_pred * y_true)
        pFP = Sum(y_pred * (1 - y_true))
        TP + FN = Sum(y_true)

    Args:
        y_true (array-like): Ground truth labels (binary 0 or 1).
        y_pred (array-like): Predicted probabilities (between 0 and 1).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The computed pF1 score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float arrays for calculation
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    # Calculate probabilistic True Positives (pTP)
    # The sum of predicted probabilities for the positive class where the actual class is positive
    p_tp = np.sum(y_pred * y_true)

    # Calculate probabilistic False Positives (pFP)
    # The sum of predicted probabilities for the positive class where the actual class is negative
    p_fp = np.sum(y_pred * (1 - y_true))

    # Calculate Total Positives (TP + FN)
    # The total count of actual positive cases
    total_positives = np.sum(y_true)

    # Calculate pPrecision
    # pPrecision = pTP / (pTP + pFP)
    # Note: The denominator (pTP + pFP) is mathematically equivalent to sum(y_pred)
    denominator_precision = p_tp + p_fp
    p_precision = p_tp / (denominator_precision + epsilon)

    # Calculate pRecall
    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # Calculate pF1
    # Harmonic mean of pPrecision and pRecall
    denominator_f1 = p_precision + p_recall
    p_f1 = 2 * (p_precision * p_recall) / (denominator_f1 + epsilon)

    return float(p_f1)
