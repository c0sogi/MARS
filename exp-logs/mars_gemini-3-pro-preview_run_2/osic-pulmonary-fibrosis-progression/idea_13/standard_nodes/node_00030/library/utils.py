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
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (array-like): Ground truth FVC values.
        y_pred (array-like): Predicted FVC values.
        sigma (array-like): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score over the input arrays.
    """
    # Ensure inputs are numpy arrays of float type
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Clip sigma to a minimum of 70 ml
    sigma_clipped = np.maximum(sigma, 70)

    # Calculate absolute error and clip at 1000 ml
    absolute_error = np.abs(y_true - y_pred)
    delta = np.minimum(absolute_error, 1000)

    # Calculate the metric term
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = np.log(sqrt_2 * sigma_clipped)

    metric = -term1 - term2

    # Return the mean score
    return np.mean(metric)
