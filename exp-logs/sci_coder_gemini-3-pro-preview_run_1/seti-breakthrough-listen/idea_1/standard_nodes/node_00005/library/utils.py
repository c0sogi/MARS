import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
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
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred) -> float:
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    Handles both numpy arrays and torch tensors.

    Args:
        y_true (np.ndarray or torch.Tensor): True binary labels.
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check if we have both classes to avoid ValueError in roc_auc_score
    # In a proper validation loop, this should rarely happen if batch/set size is sufficient
    if len(np.unique(y_true)) < 2:
        # Fallback or let it raise error depending on preference.
        # Standard behavior is to raise ValueError, but for robustness in small debug batches:
        return 0.5

    return roc_auc_score(y_true, y_pred)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
