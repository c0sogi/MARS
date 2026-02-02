import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed: The integer seed to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic algorithms can be slower but ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true, y_pred, beta: float = 1.0, epsilon: float = 1e-7) -> float:
    """
    Calculates the Probabilistic F1 score (pF1).

    This metric extends the traditional F-score to accept probabilities instead of
    binary classifications.

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred: Array-like of predicted probabilities (0 to 1).
        beta: Weight of recall in the F-score. Default is 1.0.
        epsilon: Small constant to prevent division by zero.

    Returns:
        The pF1 score as a float.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays of floats
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Flatten arrays to ensure 1D alignment
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Calculate Probabilistic True Positives (pTP)
    # pTP = Sum(y_true * y_pred)
    p_tp = np.sum(y_true * y_pred)

    # Calculate Probabilistic False Positives (pFP)
    # pFP = Sum((1 - y_true) * y_pred)
    p_fp = np.sum((1 - y_true) * y_pred)

    # Calculate Support (Actual Positives: TP + FN)
    actual_positives = np.sum(y_true)

    # Calculate pPrecision
    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP = Sum(y_pred)
    total_predicted_prob = p_tp + p_fp
    p_precision = p_tp / (total_predicted_prob + epsilon)

    # Calculate pRecall
    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (actual_positives + epsilon)

    # Calculate pF1 (Harmonic mean of pPrecision and pRecall)
    # F_beta = (1 + beta^2) * (Precision * Recall) / ((beta^2 * Precision) + Recall)
    beta_sq = beta**2
    numerator = (1 + beta_sq) * (p_precision * p_recall)
    denominator = (beta_sq * p_precision) + p_recall

    pf1 = numerator / (denominator + epsilon)

    return float(pf1)
