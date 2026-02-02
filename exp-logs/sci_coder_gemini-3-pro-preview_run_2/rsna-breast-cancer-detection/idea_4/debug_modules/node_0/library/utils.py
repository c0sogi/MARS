import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def probabilistic_f1(y_true, y_pred, beta=1.0, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    The metric is an extension of the traditional F-score that accepts probabilities
    instead of binary classifications.

    Formula:
        pF1 = (1 + beta^2) * (pPrecision * pRecall) / ((beta^2 * pPrecision) + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)
        pTP = sum(y_pred * y_true)
        pFP = sum(y_pred * (1 - y_true))
        TP + FN = sum(y_true)

    Args:
        y_true: Ground truth labels (0 or 1). Can be List, Numpy array, or Torch tensor.
        y_pred: Predicted probabilities in range [0, 1]. Can be List, Numpy array, or Torch tensor.
        beta (float): The beta value for the F-score (default 1.0 for F1).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The calculated probabilistic F1 score.
    """
    # Convert Torch tensors to Numpy if necessary
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays of float type
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Flatten arrays to ensure 1D vectors
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Calculate Probabilistic True Positives (pTP)
    # The sum of probability mass assigned to the positive class for actual positive samples.
    p_tp = np.sum(y_pred * y_true)

    # Calculate Probabilistic False Positives (pFP)
    # The sum of probability mass assigned to the positive class for actual negative samples.
    p_fp = np.sum(y_pred * (1.0 - y_true))

    # Calculate Total Actual Positives (TP + FN)
    total_positives = np.sum(y_true)

    # Calculate Probabilistic Precision
    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP is mathematically equivalent to sum(y_pred)
    predicted_mass = p_tp + p_fp
    p_precision = p_tp / (predicted_mass + epsilon)

    # Calculate Probabilistic Recall
    # pRecall = pTP / (Total Positives)
    p_recall = p_tp / (total_positives + epsilon)

    # Calculate pF1 (Harmonic Mean)
    beta2 = beta**2
    numerator = (1 + beta2) * p_precision * p_recall
    denominator = (beta2 * p_precision) + p_recall

    pf1 = numerator / (denominator + epsilon)

    return float(pf1)
