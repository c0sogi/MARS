import os
import random
import math
import numpy as np
import torch


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
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: True FVC values (torch.Tensor or np.array)
        y_pred: Predicted FVC values (torch.Tensor or np.array)
        sigma: Predicted confidence (standard deviation) (torch.Tensor or np.array)

    Returns:
        The average metric score (scalar tensor).
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma)

    # Ensure inputs are float and on the same device
    # We use y_pred as the reference for device
    device = y_pred.device
    y_true = y_true.to(device).float()
    y_pred = y_pred.float()
    sigma = sigma.to(device).float()

    # Clipping sigma (confidence) at 70 ml
    sigma_clipped = torch.clamp(sigma, min=70)

    # Calculate absolute error
    delta = torch.abs(y_true - y_pred)

    # Clipping delta (error) at 1000 ml
    delta = torch.clamp(delta, max=1000)

    # Metric calculation
    sqrt_2 = math.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

    # Return average across the batch
    return torch.mean(metric)
