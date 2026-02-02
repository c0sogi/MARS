import numpy as np
import torch
from library.config import seed_everything


def metric_laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): Ground truth FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        sigma (np.array or torch.Tensor): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score across the batch.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float32 numpy arrays for consistent calculation
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    sigma = np.array(sigma, dtype=np.float32)

    # Constants from task description
    MIN_SIGMA = 70.0
    MAX_ERROR = 1000.0
    SQRT_2 = np.sqrt(2)

    # 1. Clip the confidence (sigma)
    sigma_clipped = np.maximum(sigma, MIN_SIGMA)

    # 2. Calculate absolute error and threshold it
    abs_error = np.abs(y_true - y_pred)
    delta = np.minimum(abs_error, MAX_ERROR)

    # 3. Compute the metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    term1 = (SQRT_2 * delta) / sigma_clipped
    term2 = np.log(SQRT_2 * sigma_clipped)
    metric = -term1 - term2

    # Return the mean score
    return float(np.mean(metric))
