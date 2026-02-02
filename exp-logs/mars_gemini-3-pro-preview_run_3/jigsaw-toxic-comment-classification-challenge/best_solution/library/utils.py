import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Computes the mean column-wise ROC AUC score.

    Args:
        y_true (np.ndarray, pd.DataFrame, or torch.Tensor): Ground truth labels.
        y_pred (np.ndarray, pd.DataFrame, or torch.Tensor): Predicted probabilities.

    Returns:
        float: The mean column-wise ROC AUC score.
    """
    # Convert PyTorch tensors to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert Pandas objects to numpy
    if hasattr(y_true, "values"):
        y_true = y_true.values
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values

    # Calculate mean column-wise ROC AUC
    # average='macro' computes the metric independently for each class and then takes the average
    return roc_auc_score(y_true, y_pred, average="macro")
