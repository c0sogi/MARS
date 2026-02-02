import math
import numpy as np
import torch
from library.config import seed_everything


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


def metric_laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (Tensor or numpy array).
        y_pred: Predicted FVC values (Tensor or numpy array).
        sigma: Predicted confidence/std values (Tensor or numpy array).

    Returns:
        float: The average metric value.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are numpy arrays for consistent broadcasting
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma = np.array(sigma)

    # Clipping sigma (confidence) to a minimum of 70 ml
    sigma_clipped = np.maximum(sigma, 70)

    # Calculate absolute error (delta)
    delta = np.abs(y_true - y_pred)

    # Clipping delta (error) to a maximum of 1000 ml
    delta = np.minimum(delta, 1000)

    # Calculate metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
