import numpy as np
import torch
from library.config import Config, setup_reproducibility


def seed_everything(seed=Config.SEED):
    """
    Sets random seeds for reproducibility.
    Delegates to the centralized setup_reproducibility function in the config.

    Args:
        seed (int): The random seed to use. Defaults to Config.SEED.
    """
    setup_reproducibility(seed)


def calculate_metric(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    The metric is computed as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth FVC values.
        y_pred (np.ndarray or torch.Tensor): Predicted FVC values.
        sigma (np.ndarray or torch.Tensor): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score across the provided batch.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float64 for precision
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    sigma = sigma.astype(np.float64)

    # 1. Clip Confidence (sigma)
    # "confidence values are clipped at 70 ml"
    sigma_clipped = np.maximum(sigma, Config.CONFIDENCE_CLIP)

    # 2. Calculate Absolute Error (Delta)
    abs_error = np.abs(y_true - y_pred)

    # 3. Clip Error
    # "The error is thresholded at 1000 ml"
    delta = np.minimum(abs_error, Config.ERR_CLIP_THRESHOLD)

    # 4. Compute Metric Terms
    sqrt_2 = np.sqrt(2)

    # Term 1: - (sqrt(2) * delta) / sigma_clipped
    term1 = -(sqrt_2 * delta) / sigma_clipped

    # Term 2: - ln(sqrt(2) * sigma_clipped)
    term2 = -np.log(sqrt_2 * sigma_clipped)

    # Combine terms
    metric_values = term1 + term2

    # Return the average metric across the batch
    return np.mean(metric_values)
