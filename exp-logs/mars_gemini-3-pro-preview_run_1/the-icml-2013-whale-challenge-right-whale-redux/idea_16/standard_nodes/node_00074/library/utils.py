import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def mixup_data(x, y, alpha=1.0, device=None):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input batch of data.
        y (torch.Tensor): Target labels.
        alpha (float): Parameter for the Beta distribution.
        device (torch.device, optional): Device to place the indices on.
                                         If None, uses x.device.

    Returns:
        mixed_x (torch.Tensor): Mixed input data.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device is None:
        device = x.device

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.

    Args:
        criterion (callable): Loss function.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): Mixing coefficient.

    Returns:
        torch.Tensor: Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores or probabilities.

    Returns:
        float: AUC score. Returns 0.5 if calculation fails (e.g., single class).
    """
    try:
        # Ensure inputs are numpy arrays for safety
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # This can happen if y_true only has one class in the current batch/set
        return 0.5
