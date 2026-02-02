import os
import random
import numpy as np
import torch
from library.config import Config


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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
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


def calculate_metric(y_true, y_pred, y_conf):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (numpy array or torch tensor).
        y_pred: Predicted FVC values (numpy array or torch tensor).
        y_conf: Predicted Confidence (sigma) values (numpy array or torch tensor).

    Returns:
        float: The average metric score across the batch.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_conf, torch.Tensor):
        y_conf = y_conf.detach().cpu().numpy()

    # Ensure inputs are numpy arrays and float type for calculation
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    y_conf = np.array(y_conf, dtype=np.float32)

    # Apply clipping to confidence (sigma)
    # sigma_clipped = max(sigma, 70)
    sigma_clipped = np.maximum(y_conf, Config.sigma_clip)

    # Calculate absolute error (delta)
    delta = np.abs(y_true - y_pred)

    # Apply thresholding to error
    # delta = min(|true - pred|, 1000)
    delta = np.minimum(delta, Config.max_error)

    # Calculate metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the average metric
    return np.mean(metric)
