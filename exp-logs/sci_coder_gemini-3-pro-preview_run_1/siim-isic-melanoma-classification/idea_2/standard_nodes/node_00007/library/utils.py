import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import SEED


def seed_everything(seed: int = SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for consistent results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred) -> float:
    """
    Computes the Area Under the ROC Curve (AUC) for validation and testing.

    Args:
        y_true: Ground truth binary labels (numpy array or torch tensor).
        y_pred: Predicted probabilities for the positive class (numpy array or torch tensor).

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flattened
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    # Handle edge case where only one class is present in the targets
    # This can happen in small batches or highly imbalanced subsets
    if len(np.unique(y_true)) < 2:
        # AUC is undefined if only one class is present.
        # Returning 0.5 (random guessing) is a safe fallback for logging,
        # though ideally validation sets should be stratified.
        return 0.5

    return roc_auc_score(y_true, y_pred)
