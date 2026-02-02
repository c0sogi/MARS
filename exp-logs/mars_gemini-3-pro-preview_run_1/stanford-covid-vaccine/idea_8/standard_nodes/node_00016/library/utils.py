import os
import random
import numpy as np
import torch


class AverageMeter:
    """Computes and stores the average and current value."""

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


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Formula: Mean(Sqrt(Mean((y_true - y_pred)^2, axis=0)))

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values. Shape (N, num_scored_cols).
        y_pred (np.ndarray or torch.Tensor): Predicted values. Shape (N, num_scored_cols).

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate MSE per column
    # axis=0 represents the sample dimension
    mse_per_col = np.mean((y_true - y_pred) ** 2, axis=0)

    # Calculate RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate Mean of RMSEs across columns
    score = np.mean(rmse_per_col)

    return float(score)
