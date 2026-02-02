import torch
import numpy as np
from library.config import seed_everything


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training.
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


def metric_laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (torch.Tensor or np.ndarray).
        y_pred: Predicted FVC values (torch.Tensor or np.ndarray).
        sigma: Predicted confidence (standard deviation) (torch.Tensor or np.ndarray).

    Returns:
        float: The average metric score across the input batch.
    """
    # Ensure inputs are torch tensors
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma)

    # Ensure float precision and flatten for element-wise operations
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)
    sigma = sigma.float().view(-1)

    # Define constants on the correct device
    device = y_true.device
    sqrt_2 = torch.tensor(2.0, device=device).sqrt()

    # Apply clipping constraints defined in the metric
    sigma_clipped = torch.clamp(sigma, min=70)
    delta = torch.abs(y_true - y_pred)
    delta = torch.clamp(delta, max=1000)

    # Calculate the metric
    # Note: The metric is negative, and higher (closer to 0) is better.
    metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

    return torch.mean(metric).item()
