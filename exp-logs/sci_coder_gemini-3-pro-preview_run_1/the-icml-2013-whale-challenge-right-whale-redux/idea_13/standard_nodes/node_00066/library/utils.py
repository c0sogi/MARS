import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
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

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auc(y_true, y_score):
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor).
        y_score: Predicted probabilities (numpy array or torch tensor).

    Returns:
        float: The AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_score, torch.Tensor):
        y_score = y_score.detach().cpu().numpy()

    # Flatten arrays to ensure 1D
    y_true = np.ravel(y_true)
    y_score = np.ravel(y_score)

    # Check for NaN or Inf
    if np.isnan(y_score).any() or np.isinf(y_score).any():
        # Replace NaN/Inf with 0 or 0.5 to avoid crash, though this indicates a model issue
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)

    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        # This usually happens if the batch contains only one class
        # Return 0.5 as a neutral score
        return 0.5


def print_metric(name, value):
    """
    Prints a metric with full precision.

    Args:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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
