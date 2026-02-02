import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Computes the Probabilistic F1 score (pF1).

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)

        Where:
        pTP = Sum(y_true * y_pred)
        pPrecision = pTP / (pTP + pFP) = pTP / Sum(y_pred)
        pRecall = pTP / (TP + FN) = pTP / Sum(y_true)

    Args:
        y_true (array-like or Tensor): Ground truth binary labels (0 or 1).
        y_pred (array-like or Tensor): Predicted probabilities (0 to 1).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The pF1 score.
    """
    # Handle PyTorch Tensors: detach and move to CPU if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Probabilistic True Positives (pTP)
    # The intersection of the prediction probability and the ground truth
    p_tp = np.sum(y_true * y_pred)

    # Probabilistic Precision
    # Denominator is pTP + pFP, which simplifies to the sum of predicted probabilities
    p_precision_denom = np.sum(y_pred)
    p_precision = p_tp / (p_precision_denom + epsilon)

    # Probabilistic Recall
    # Denominator is TP + FN, which is the total count of actual positive labels
    p_recall_denom = np.sum(y_true)
    p_recall = p_tp / (p_recall_denom + epsilon)

    # Probabilistic F1
    # Harmonic mean of pPrecision and pRecall
    f1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return float(f1)
