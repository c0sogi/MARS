import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise ROC AUC score.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (one-hot encoded).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate ROC AUC
    # average='macro' computes the metric for each label, and finds their unweighted mean.
    # This corresponds to "Mean column-wise ROC AUC".
    try:
        metric = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # This handles cases where a specific class might not be present in the y_true batch
        # preventing the code from crashing during validation steps on small batches.
        metric = 0.0

    return metric
