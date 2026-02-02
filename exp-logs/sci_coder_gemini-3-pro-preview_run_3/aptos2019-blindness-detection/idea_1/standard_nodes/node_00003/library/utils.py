import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.

    Handles the conversion of continuous regression outputs to integer class labels
    before calculating the metric.

    Args:
        y_true (np.array or list): Ground truth labels (integers 0-4).
        y_pred (np.array or list): Predicted values. Can be continuous (from regression)
                                   or integers.

    Returns:
        float: The Quadratic Weighted Kappa score.
    """
    # Convert inputs to numpy arrays if they aren't already
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Post-processing for regression outputs:
    # 1. Clip values to valid range [0, 4] to handle outliers
    y_pred_clipped = np.clip(y_pred_np, 0, 4)

    # 2. Round to nearest integer to get class labels
    y_pred_rounded = np.round(y_pred_clipped).astype(int)

    # Ensure ground truth is integer type
    y_true_int = y_true_np.astype(int)

    # Calculate Quadratic Weighted Kappa
    # weights='quadratic' ensures the penalty is squared based on distance
    score = cohen_kappa_score(y_true_int, y_pred_rounded, weights="quadratic")

    return score
