import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (ROC AUC) for multi-label classification.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (binary).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check for NaN in predictions before metric calculation
    if np.isnan(y_pred).any():
        raise ValueError("Predictions contain NaN values, cannot calculate ROC AUC.")

    try:
        # 'macro' average calculates metrics for each label, and finds their unweighted mean.
        # This is standard for multi-label classification tasks.
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # This can happen if a class has only one label present in the set (all 0s or all 1s)
        # In a robust validation set this shouldn't happen, but we handle it gracefully.
        score = 0.5

    return score


def check_tensor_sanitation(data, name="Tensor"):
    """
    Explicitly validates that the input data contains no NaNs or Infinite values.
    This is crucial for the 'Sanitized' pipeline to prevent data corruption,
    especially when dealing with pseudo-labels.

    Args:
        data (torch.Tensor or np.ndarray): The data to check.
        name (str): Name of the variable for the error message.

    Raises:
        ValueError: If NaNs or Infs are detected.

    Returns:
        bool: True if sanitary.
    """
    if isinstance(data, torch.Tensor):
        has_nan = torch.isnan(data).any().item()
        has_inf = torch.isinf(data).any().item()
    elif isinstance(data, np.ndarray):
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
    else:
        # Try to convert to numpy for generic check if it's a list or other iterable
        try:
            arr = np.array(data)
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
        except Exception:
            # If conversion fails, assume it's not a numeric tensor/array we can check
            return True

    if has_nan:
        raise ValueError(f"Sanitation Check Failed: '{name}' contains NaN values.")

    if has_inf:
        raise ValueError(f"Sanitation Check Failed: '{name}' contains Infinite values.")

    return True
