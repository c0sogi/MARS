import os
import random
import numpy as np
import torch
from library.config import Config


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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def score(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or list): Ground truth FVC values.
        y_pred (np.array or list): Predicted FVC values.
        sigma (np.array or list): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Retrieve constants from Config
    max_error = float(Config.MAX_ERROR)
    min_confidence = float(Config.MIN_CONFIDENCE)

    # Clip confidence (sigma)
    sigma_clipped = np.maximum(sigma, min_confidence)

    # Calculate absolute error
    abs_error = np.abs(y_true - y_pred)

    # Clip error (delta)
    delta = np.minimum(abs_error, max_error)

    # Compute metric term by term
    # Term 1: - (sqrt(2) * delta) / sigma_clipped
    term1 = -(np.sqrt(2) * delta) / sigma_clipped

    # Term 2: - ln(sqrt(2) * sigma_clipped)
    term2 = -np.log(np.sqrt(2) * sigma_clipped)

    metric = term1 + term2

    return np.mean(metric)
