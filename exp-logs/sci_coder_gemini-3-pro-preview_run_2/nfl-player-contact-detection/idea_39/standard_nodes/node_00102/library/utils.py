import os
import random
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef


def seed_everything(seed: int):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

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


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth binary labels.
        y_pred (np.ndarray or torch.Tensor): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flat
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    return matthews_corrcoef(y_true, y_pred)
