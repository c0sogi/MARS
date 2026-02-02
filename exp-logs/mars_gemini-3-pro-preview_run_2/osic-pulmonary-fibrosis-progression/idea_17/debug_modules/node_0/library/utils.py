import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def score_func(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        score = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or list): Ground truth FVC values.
        y_pred (np.array or list): Predicted FVC values.
        sigma (np.array or list): Predicted confidence (standard deviation).

    Returns:
        float: The mean metric score over the input samples.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Constants from metric definition
    MIN_CONFIDENCE = 70
    MAX_ERROR = 1000

    # Clip confidence values to avoid singularities and reflect measurement uncertainty
    sigma_clipped = np.maximum(sigma, MIN_CONFIDENCE)

    # Calculate absolute error and threshold it to avoid excessive penalties for outliers
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, MAX_ERROR)

    # Calculate the metric components
    # Term 1: Scaled error
    term1 = (np.sqrt(2) * delta) / sigma_clipped

    # Term 2: Log of scaled uncertainty
    term2 = np.log(np.sqrt(2) * sigma_clipped)

    # Combine terms
    metric = -term1 - term2

    # Return the average score
    return np.mean(metric)
