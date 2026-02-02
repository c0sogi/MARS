import os
import random
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
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

    # Enforce deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred) -> float:
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (binary). Can be list, numpy array, or torch Tensor.
        y_pred: Predicted probabilities. Can be list, numpy array, or torch Tensor.

    Returns:
        float: The ROC AUC score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    try:
        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Handle edge case where y_true has only one class (e.g., in a small batch)
        # Return 0.5 as a neutral baseline
        score = 0.5

    return score


def print_metric(name: str, value: float) -> None:
    """
    Prints a metric with its full precision without rounding.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")
    sys.stdout.flush()
