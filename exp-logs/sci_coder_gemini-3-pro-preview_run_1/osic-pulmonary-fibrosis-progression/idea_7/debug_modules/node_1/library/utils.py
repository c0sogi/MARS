import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric for the competition.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): Ground truth FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        sigma (np.array or torch.Tensor): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score across the batch.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are numpy arrays (handles lists, etc.)
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Apply clipping as defined in the task and Config
    sigma_clipped = np.maximum(sigma, Config.MIN_CONFIDENCE_CLIP)

    absolute_error = np.abs(y_true - y_pred)
    delta = np.minimum(absolute_error, Config.MAX_ERROR_CLIP)

    # Calculate metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric_values)
