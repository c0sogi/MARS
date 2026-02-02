import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import seed_everything, DEVICE


def get_device():
    """
    Returns the PyTorch device configured in the config file.
    """
    return torch.device(DEVICE)


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


def compute_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (0 or 1). Can be Tensor or numpy array.
        y_pred: Predicted probabilities for class 1. Can be Tensor or numpy array.

    Returns:
        float: ROC AUC score. Returns 0.5 if only one class is present in y_true.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to ensure 1D
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Check for single class case (cannot compute ROC AUC)
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def print_metrics(metrics_dict):
    """
    Prints metric values with full precision as required.

    Args:
        metrics_dict: Dictionary of metric names and values.
    """
    print("Validation Metrics:")
    for key, value in metrics_dict.items():
        print(f"{key}: {value}")
