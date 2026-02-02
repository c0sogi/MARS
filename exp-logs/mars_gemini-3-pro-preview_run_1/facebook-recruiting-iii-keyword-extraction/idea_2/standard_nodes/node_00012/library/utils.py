import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_f1_score(y_true, y_pred, average: str = "samples") -> float:
    """
    Calculates the F1 score for multi-label classification.

    Handles both numpy arrays and torch tensors.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels.
        y_pred (np.ndarray or torch.Tensor): Predicted binary labels (0 or 1).
        average (str): The averaging strategy for F1 score.
                       Options: 'micro', 'macro', 'samples', 'weighted'.
                       Defaults to 'samples' (Mean F1-Score per instance).

    Returns:
        float: The calculated F1 score.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are integers (binary)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Calculate F1 score using scikit-learn
    # zero_division=0 prevents warnings/errors when a label is not present in the batch
    score = f1_score(y_true, y_pred, average=average, zero_division=0)

    return float(score)
