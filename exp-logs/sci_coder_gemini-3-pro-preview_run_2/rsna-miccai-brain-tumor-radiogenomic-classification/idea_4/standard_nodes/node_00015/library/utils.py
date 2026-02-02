import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred: Array-like of predicted probabilities for class 1.

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present in y_true
               to avoid exceptions during edge cases.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if we have both classes to calculate AUC
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
    """

    def __init__(self, name="Metric"):
        self.name = name
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


def print_training_status(epoch, total_epochs, batch_idx, total_batches, loss_meter):
    """
    Helper function to log training progress to console.
    Prints full float precision.
    """
    print(
        f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx}/{total_batches}] Loss: {loss_meter.val} (Avg: {loss_meter.avg})"
    )


def print_validation_metric(metric_name, value):
    """
    Helper to print validation metrics with full precision.
    """
    print(f"Validation {metric_name}: {value}")
