import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_weighted_log_loss_score(y_pred, y_true, epsilon=1e-15):
    """
    Calculates the weighted multi-label logarithmic loss for the competition.

    The metric is calculated as the average of the weighted binary cross entropy
    for each label. The labels are C1-C7 and patient_overall.

    Weights:
        C1-C7: 1.0
        patient_overall: 7.0

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities.
            Shape should be (N, 7) or (N, 8).
            If (N, 7), it is assumed to be C1-C7, and patient_overall is derived as max(C1-C7).
            If (N, 8), columns are assumed to be [C1, C2, C3, C4, C5, C6, C7, patient_overall].
        y_true (torch.Tensor or np.ndarray): Ground truth labels.
            Shape should be (N, 8).
            Columns: [C1, C2, C3, C4, C5, C6, C7, patient_overall].
        epsilon (float): Small constant to avoid log(0).

    Returns:
        float: The weighted log loss score.
    """
    # Convert tensors to numpy
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Handle prediction shape
    # If model predicts only C1-C7, derive patient_overall
    if y_pred.shape[1] == 7:
        # Derive patient_overall as the max probability of any fracture
        p_overall = np.max(y_pred, axis=1, keepdims=True)
        y_pred = np.concatenate([y_pred, p_overall], axis=1)

    # Validation checks
    if y_pred.shape[1] != 8:
        raise ValueError(
            f"y_pred must have 8 columns (or 7 to derive the 8th). Got {y_pred.shape[1]}."
        )
    if y_true.shape[1] != 8:
        raise ValueError(f"y_true must have 8 columns. Got {y_true.shape[1]}.")

    # Clip predictions for numerical stability
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Define weights
    # C1-C7 have weight 1.0
    # patient_overall has weight 7.0 (Weighted more highly)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])

    # Calculate weighted Log Loss
    # Formula: -w * [y * log(p) + (1-y) * log(1-p)]
    # We use broadcasting for weights
    loss_matrix = -weights * (
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

    # The metric is the average loss across all rows (flattened)
    return np.mean(loss_matrix)
