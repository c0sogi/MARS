import torch
import numpy as np
from library.config import Config, seed_everything


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the competition.

    This metric is used for both evaluation and loss calculation.
    Metric = - (sqrt(2) * delta_clipped) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    where:
        delta_clipped = min(|y_true - y_pred|, 1000)
        sigma_clipped = max(sigma, 70)

    Args:
        y_true: True FVC values (Tensor or numpy array).
        y_pred: Predicted FVC values (Tensor or numpy array).
        sigma: Predicted confidence (sigma) values (Tensor or numpy array).

    Returns:
        torch.Tensor: The average metric score (scalar). Values are negative, higher is better.
    """
    # Convert numpy arrays to tensors if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if isinstance(sigma, np.ndarray):
        sigma = torch.tensor(sigma, dtype=torch.float32)

    # Ensure inputs are float tensors
    y_true = y_true.float()
    y_pred = y_pred.float()
    sigma = sigma.float()

    # Flatten tensors to ensure element-wise operations align correctly
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    sigma = sigma.view(-1)

    # Define constants on the correct device
    device = y_true.device
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))

    # Apply clipping logic defined in the metric
    # Confidence clipped at 70 ml
    sigma_clipped = torch.clamp(sigma, min=Config.MIN_CONFIDENCE)

    # Absolute error
    delta = torch.abs(y_true - y_pred)

    # Error thresholded at 1000 ml
    delta_clipped = torch.clamp(delta, max=Config.MAX_ERROR)

    # Calculate metric formula
    # metric = - (sqrt(2) * delta / sigma) - ln(sqrt(2) * sigma)
    metric = -(sqrt_2 * delta_clipped) / sigma_clipped - torch.log(
        sqrt_2 * sigma_clipped
    )

    # Return the mean metric over the batch
    return torch.mean(metric)
