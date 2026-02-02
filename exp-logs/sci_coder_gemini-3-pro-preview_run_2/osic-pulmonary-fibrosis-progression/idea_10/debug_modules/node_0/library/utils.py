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


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Metric = - (sqrt(2) * Delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Where:
    sigma_clipped = max(sigma, 70)
    Delta = min(|y_true - y_pred|, 1000)

    Args:
        y_true (np.array or list): True FVC values.
        y_pred (np.array or list): Predicted FVC values.
        sigma (np.array or list): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score (higher is better, usually negative).
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Constants from Config
    max_error = Config.MAX_ERROR
    min_uncertainty = Config.MIN_UNCERTAINTY

    # Clip sigma to avoid infinite penalties for very small uncertainty
    sigma_clipped = np.maximum(sigma, min_uncertainty)

    # Calculate absolute error and clip it to avoid excessive penalties for outliers
    abs_error = np.abs(y_true - y_pred)
    delta = np.minimum(abs_error, max_error)

    # Calculate metric terms
    # Term 1: Scaled error
    term1 = -(np.sqrt(2) * delta) / sigma_clipped
    # Term 2: Log of the normalization constant
    term2 = -np.log(np.sqrt(2) * sigma_clipped)

    metric = term1 + term2

    return np.mean(metric)


def mad_to_sigma(mad):
    """
    Converts Mean Absolute Deviation (MAD) to the Laplace scale parameter Sigma.

    For a Laplace distribution, the standard deviation (sigma) is related to the
    scale parameter (b) by sigma = sqrt(2) * b.
    Since MAD is the maximum likelihood estimator for b in Laplace regression,
    we scale it by sqrt(2) to get the standard deviation expected by the metric.

    Args:
        mad (np.array or float): Predicted Mean Absolute Deviation.

    Returns:
        np.array or float: Converted Sigma (Confidence).
    """
    return mad * np.sqrt(2)
