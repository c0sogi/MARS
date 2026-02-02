import numpy as np
import torch
from library.config import seed_everything


def calculate_metric(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (numpy array, list, or torch.Tensor).
        y_pred: Predicted FVC values (numpy array, list, or torch.Tensor).
        sigma: Predicted confidence (standard deviation) values (numpy array, list, or torch.Tensor).

    Returns:
        float: The average metric score across all samples.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float numpy arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Constants defined in the metric
    MAX_ERROR = 1000.0
    MIN_CONFIDENCE = 70.0
    SQRT_2 = np.sqrt(2)

    # Clip confidence values
    sigma_clipped = np.maximum(sigma, MIN_CONFIDENCE)

    # Calculate absolute error and clip it
    abs_diff = np.abs(y_true - y_pred)
    delta = np.minimum(abs_diff, MAX_ERROR)

    # Calculate metric
    term1 = (SQRT_2 * delta) / sigma_clipped
    term2 = np.log(SQRT_2 * sigma_clipped)
    metric = -term1 - term2

    return np.mean(metric)
