import numpy as np
import torch
from library.config import seed_everything


def score_function(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric for Lung Function Prediction.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: True FVC values (numpy array, torch tensor, or list)
        y_pred: Predicted FVC values (numpy array, torch tensor, or list)
        sigma: Predicted Confidence (sigma) values (numpy array, torch tensor, or list)

    Returns:
        float: The average metric score across all samples.
    """
    # Convert torch tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    sigma = np.array(sigma, dtype=np.float32)

    # Define constants based on metric definition
    SIGMA_CLIP = 70.0
    DELTA_CLIP = 1000.0
    SQRT_2 = np.sqrt(2)

    # 1. Clip the confidence values (sigma)
    sigma_clipped = np.maximum(sigma, SIGMA_CLIP)

    # 2. Calculate absolute error (delta) and clip it
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, DELTA_CLIP)

    # 3. Compute the metric
    # Formula: - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    metric = -(SQRT_2 * delta) / sigma_clipped - np.log(SQRT_2 * sigma_clipped)

    # Return the mean metric across the batch
    return np.mean(metric)
