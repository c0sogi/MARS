import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true, y_pred, beta=1.0, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1) as defined in the task.

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    pPrecision = pTP / (pTP + pFP)
    pRecall = pTP / (TP + FN)

    Args:
        y_true: Array-like or Tensor of ground truth labels (0 or 1).
        y_pred: Array-like or Tensor of predicted probabilities (0 to 1).
        beta (float): Weight of precision in harmonic mean (default 1.0).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The pF1 score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Shape consistency check
    if y_true.shape != y_pred.shape:
        # Attempt to squeeze if one dimension is 1 (common in PyTorch outputs)
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )

    # Calculate Probabilistic True Positives (pTP)
    # pTP = Sum(Prediction_Probability * Ground_Truth)
    p_tp = np.sum(y_pred * y_true)

    # Calculate Probabilistic False Positives (pFP)
    # pFP = Sum(Prediction_Probability * (1 - Ground_Truth))
    p_fp = np.sum(y_pred * (1 - y_true))

    # Calculate Total Positives (TP + FN)
    # This is the count of actual positive cases in the ground truth
    total_positives = np.sum(y_true)

    # Calculate Probabilistic Precision
    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP is mathematically equal to Sum(y_pred)
    sum_predictions = p_tp + p_fp
    p_precision = p_tp / (sum_predictions + epsilon)

    # Calculate Probabilistic Recall
    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # Calculate pF1 Score
    # F_beta = (1 + beta^2) * (Precision * Recall) / ((beta^2 * Precision) + Recall)
    beta_sq = beta**2
    numerator = (1 + beta_sq) * p_precision * p_recall
    denominator = (beta_sq * p_precision) + p_recall

    pf1 = numerator / (denominator + epsilon)

    return pf1
