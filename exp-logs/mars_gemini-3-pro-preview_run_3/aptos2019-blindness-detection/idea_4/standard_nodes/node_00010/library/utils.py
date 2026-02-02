import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa metric for Diabetic Retinopathy detection.

    This function handles both integer labels and continuous regression outputs.
    Regression outputs are rounded to the nearest integer and clipped to the [0, 4] range.

    Args:
        y_true: Array-like or Tensor of ground truth labels (integers 0-4).
        y_pred: Array-like or Tensor of predicted scores. Can be continuous (regression output)
                or integers.

    Returns:
        float: The Quadratic Weighted Kappa score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Process predictions:
    # 1. Round to nearest integer (for regression outputs)
    # 2. Clip to valid range [0, 4]
    # 3. Cast to integer type
    y_pred = np.clip(np.round(y_pred), 0, 4).astype(int)

    # Ensure ground truth is integer type
    y_true = y_true.astype(int)

    # Calculate Cohen's Kappa with quadratic weights
    # We explicitly provide labels=[0, 1, 2, 3, 4] to ensure the confusion matrix
    # is always 5x5, even if the current batch/set doesn't contain all classes.
    return cohen_kappa_score(
        y_true, y_pred, weights="quadratic", labels=[0, 1, 2, 3, 4]
    )
