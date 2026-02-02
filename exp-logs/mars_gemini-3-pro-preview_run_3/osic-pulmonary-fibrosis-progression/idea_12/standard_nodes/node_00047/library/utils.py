import os
import random
import numpy as np
import torch
from library.config import MIN_UNCERTAINTY, MAX_ERROR


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred, sigma_pred):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): True FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        sigma_pred (np.array or torch.Tensor): Predicted confidence (sigma).

    Returns:
        float: The average metric score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma_pred, torch.Tensor):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    # Ensure inputs are float for calculation
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    sigma_pred = sigma_pred.astype(float)

    # Clip sigma to reflect approximate measurement uncertainty
    sigma_clipped = np.maximum(sigma_pred, MIN_UNCERTAINTY)

    # Calculate absolute error
    delta = np.abs(y_true - y_pred)

    # Threshold error to avoid large errors adversely penalizing results
    delta = np.minimum(delta, MAX_ERROR)

    # Calculate metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
