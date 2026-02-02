import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate torch device (cuda or cpu).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def compute_metric(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor): True FVC values.
        y_pred (torch.Tensor): Predicted FVC values.
        sigma (torch.Tensor): Predicted confidence (std dev).

    Returns:
        torch.Tensor: Scalar metric value (average across batch).
    """
    # Ensure inputs are on the same device and float
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma)

    # Clip sigma to minimum 70
    sigma_clipped = torch.clamp(sigma, min=70)

    # Calculate absolute error and clip to 1000
    delta = torch.abs(y_true - y_pred)
    delta = torch.clamp(delta, max=1000)

    # Calculate metric
    # Use torch.tensor for sqrt(2) to ensure device compatibility
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=y_true.device))

    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = torch.log(sqrt_2 * sigma_clipped)

    metric = -term1 - term2

    return torch.mean(metric)
