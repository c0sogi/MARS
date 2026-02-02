import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
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


def get_device():
    """
    Returns the available device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The ROC AUC score. Returns 0.5 if only one class is present.
    """
    # Convert to numpy arrays if they are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Handle edge case where only one class is present in y_true
    # ROC AUC is undefined in this case, returning 0.5 is a neutral fallback
    if len(np.unique(y_true)) < 2:
        return 0.5

    return roc_auc_score(y_true, y_pred)


def print_metric(name, value):
    """
    Prints a metric with full precision as requested.

    Args:
        name (str): Name of the metric.
        value (float): Value of the metric.
    """
    print(f"{name}: {value}")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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


def save_submission(ids, predictions, path):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (list or array): List of BraTS21IDs.
        predictions (list or array): List of predicted probabilities (MGMT_value).
        path (str): File path to save the CSV.
    """
    # Ensure inputs are flat lists/arrays
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy().flatten()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy().flatten()

    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save to CSV
    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
