import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
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
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metric(true_fvc, pred_fvc, pred_sigma):
    """
    Computes the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true_fvc - pred_fvc|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        true_fvc (np.array): Ground truth FVC values.
        pred_fvc (np.array): Predicted FVC values.
        pred_sigma (np.array): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score across the input arrays.
    """
    # Ensure inputs are numpy arrays
    true_fvc = np.array(true_fvc, dtype=np.float64)
    pred_fvc = np.array(pred_fvc, dtype=np.float64)
    pred_sigma = np.array(pred_sigma, dtype=np.float64)

    # Clip confidence values (sigma)
    sigma_clipped = np.maximum(pred_sigma, Config.CONFIDENCE_MIN_THRESHOLD)

    # Calculate absolute error and clip it
    delta = np.abs(true_fvc - pred_fvc)
    delta = np.minimum(delta, Config.ERROR_MAX_THRESHOLD)

    # Calculate metric components
    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the mean metric
    return np.mean(metric_values)
