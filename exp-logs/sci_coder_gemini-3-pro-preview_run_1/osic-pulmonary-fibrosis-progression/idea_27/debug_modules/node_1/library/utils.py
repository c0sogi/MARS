import torch
import numpy as np
from library.config import seed_everything


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


def calculate_metric(y_true, y_pred, y_std):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (numpy array or torch tensor).
        y_pred: Predicted FVC values (numpy array or torch tensor).
        y_std: Predicted confidence (sigma) values (numpy array or torch tensor).

    Returns:
        The average metric score (float).
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_std, torch.Tensor):
        y_std = y_std.detach().cpu().numpy()

    # Ensure inputs are float for precision
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    y_std = y_std.astype(np.float64)

    # Calculate Delta with clipping at 1000
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, 1000.0)

    # Clip confidence (sigma) at 70
    sigma_clipped = np.maximum(y_std, 70.0)

    # Calculate metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the average score
    return np.mean(metric)
