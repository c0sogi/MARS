import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: True FVC values (array-like or tensor).
        y_pred: Predicted FVC values (array-like or tensor).
        sigma: Predicted Confidence (sigma) values (array-like or tensor).

    Returns:
        float: The average metric score (negative value, higher is better).
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(sigma):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # 1. Clip sigma
    # sigma_clipped = max(sigma, 70)
    sigma_clipped = np.maximum(sigma, Config.SIGMA_CLIP)

    # 2. Calculate delta and clip
    # delta = min(|FVC_true - FVC_pred|, 1000)
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, Config.ERROR_MAX)

    # 3. Compute metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return average across the batch/set
    return np.mean(metric)
