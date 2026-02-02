import os
import random
import numpy as np
import torch
from library.config import SEED, MIN_CONFIDENCE, MAX_ERROR


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (array-like): True FVC values.
        y_pred (array-like): Predicted FVC values.
        sigma (array-like): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score (higher is better, values are negative).
    """
    # Convert inputs to numpy arrays to ensure vectorization works
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma = np.array(sigma)

    # Clip sigma to reflect approximate measurement uncertainty (min 70 ml)
    sigma_clipped = np.maximum(sigma, MIN_CONFIDENCE)

    # Calculate absolute error
    delta = np.abs(y_true - y_pred)

    # Clip error to avoid large errors adversely penalizing results (max 1000 ml)
    delta_clipped = np.minimum(delta, MAX_ERROR)

    # Calculate the metric for each sample
    # metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
    metric = -(np.sqrt(2) * delta_clipped) / sigma_clipped - np.log(
        np.sqrt(2) * sigma_clipped
    )

    # Return the mean score
    return np.mean(metric)
