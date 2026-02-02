import numpy as np
import torch
from library.config import seed_everything


def score_function(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (numpy array, list, or torch.Tensor).
        y_pred: Predicted FVC values (numpy array, list, or torch.Tensor).
        sigma: Predicted Confidence/Std Dev values (numpy array, list, or torch.Tensor).

    Returns:
        float: The average metric score (negative value, higher is better).
    """
    # Convert tensors to numpy arrays if necessary
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

    # 1. Clip confidence values at 70 ml
    sigma_clipped = np.maximum(sigma, 70)

    # 2. Calculate absolute error and clip at 1000 ml
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, 1000)

    # 3. Compute metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the average score
    return np.mean(metric)
