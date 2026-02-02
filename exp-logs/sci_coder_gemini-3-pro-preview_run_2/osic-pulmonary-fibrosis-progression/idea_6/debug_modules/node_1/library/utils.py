import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metric(
    y_true, y_pred, sigma, clip_min=Config.CONFIDENCE_CLIP_MIN, error_max=1000
):
    """
    Computes the modified Laplace Log Likelihood metric as defined in the competition.

    Formula:
        sigma_clipped = max(sigma, clip_min)
        delta = min(|y_true - y_pred|, error_max)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or list): True FVC values.
        y_pred (np.array or list): Predicted FVC values.
        sigma (np.array or list): Predicted confidence (standard deviation).
        clip_min (float): Minimum value to clip sigma (default: 70).
        error_max (float): Maximum value to clip absolute error (default: 1000).

    Returns:
        float: The mean metric score across all samples.
    """
    # Ensure inputs are numpy arrays of float type for precision
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Clip confidence values (sigma)
    sigma_clipped = np.maximum(sigma, clip_min)

    # Calculate absolute error (delta) and clip it
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, error_max)

    # Calculate the metric
    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the mean score
    return np.mean(metric_values)
