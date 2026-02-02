import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels.
            Can be one-hot encoded (N, C) or class indices (N,).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.
            Shape (N, C).

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Check if y_true is 1D (indices) and convert to one-hot if necessary
    # This handles cases where the loss function uses indices but metric needs one-hot
    if y_true.ndim == 1 and y_pred.ndim == 2:
        num_classes = y_pred.shape[1]
        y_true_one_hot = np.zeros_like(y_pred)
        # Use np.arange to index rows and y_true for columns
        # Ensure y_true are integers for indexing
        y_true_indices = y_true.astype(int)
        y_true_one_hot[np.arange(len(y_true)), y_true_indices] = 1
        y_true = y_true_one_hot

    # Calculate ROC AUC
    # average='macro' computes the metric for each label, and finds their unweighted mean.
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # This can happen if a class is not present in the batch (e.g. only one class in y_true)
        # In this case, ROC AUC is undefined. returning 0.5 is a neutral fallback,
        # but usually this implies a bad batch or split.
        score = 0.5

    return score
