import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa metric for diabetic retinopathy severity.

    This function handles the conversion of continuous regression outputs to the
    discrete 0-4 scale required by the metric.

    Args:
        y_true: Array-like of ground truth labels. Can be a List, NumPy array, or Torch tensor.
        y_pred: Array-like of predicted scores. Can be continuous floats (regression output).
                Can be a List, NumPy array, or Torch tensor.

    Returns:
        float: The quadratic weighted kappa score.
    """
    # Detach and convert to numpy if inputs are torch tensors
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flat numpy arrays
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    # Post-process regression predictions:
    # 1. Clip values to the valid range [0, 4]
    # 2. Round to the nearest integer
    # 3. Cast to integer type
    y_pred_processed = np.clip(y_pred, 0, 4)
    y_pred_processed = np.round(y_pred_processed).astype(int)

    # Ensure ground truth is integer type
    y_true = y_true.astype(int)

    # Calculate Cohen's Kappa with quadratic weights
    score = cohen_kappa_score(y_true, y_pred_processed, weights="quadratic")

    return score
