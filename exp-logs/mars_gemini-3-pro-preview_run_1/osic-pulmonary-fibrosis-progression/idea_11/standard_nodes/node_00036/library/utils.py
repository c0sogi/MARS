import os
import random
import numpy as np
import torch
from library.config import Training, setup_reproducibility


def seed_everything(seed=Training.SEED):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
    Wraps the existing setup_reproducibility function from config.
    """
    setup_reproducibility(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
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


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.ndarray, torch.Tensor, or float): Ground truth FVC values.
        y_pred (np.ndarray, torch.Tensor, or float): Predicted FVC values.
        sigma (np.ndarray, torch.Tensor, or float): Predicted Confidence (standard deviation).

    Returns:
        float: The average metric score across the input batch.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float64 numpy arrays for precision
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # 1. Clip the confidence values (sigma)
    # The metric clips confidence at 70 ml to reflect approximate measurement uncertainty
    sigma_clipped = np.maximum(sigma, Training.MIN_SIGMA_CLIP)

    # 2. Calculate the absolute error (delta) and clip it
    # The error is thresholded at 1000 ml to avoid large errors adversely penalizing results
    abs_error = np.abs(y_true - y_pred)
    delta = np.minimum(abs_error, Training.MAX_ERROR_CLIP)

    # 3. Compute the metric
    # metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta / sigma_clipped) - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
