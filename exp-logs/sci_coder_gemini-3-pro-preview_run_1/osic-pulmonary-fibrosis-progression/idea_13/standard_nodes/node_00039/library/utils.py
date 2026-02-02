import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): True FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        sigma (np.array or torch.Tensor): Predicted Confidence (standard deviation).

    Returns:
        float: The average metric value across the batch.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float for calculation
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Calculate sigma_clipped: max(sigma, 70)
    sigma_clipped = np.maximum(sigma, 70)

    # Calculate delta: min(|FVC_true - FVC_pred|, 1000)
    absolute_error = np.abs(y_true - y_pred)
    delta = np.minimum(absolute_error, 1000)

    # Calculate metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the average score
    return np.mean(metric)
