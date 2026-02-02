import os
import random
import numpy as np
import torch
from library.config import set_seed, laplace_log_likelihood


def seed_everything(seed=42):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The random seed value.
    """
    # Use the provided library function for basic seeding (random, numpy, os environment)
    set_seed(seed)

    # Extend with PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Calculates the competition evaluation metric: Modified Laplace Log Likelihood.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or pd.Series): Ground truth FVC values.
        y_pred (np.array or pd.Series): Predicted FVC values.
        sigma (np.array or pd.Series): Predicted confidence (sigma) values.

    Returns:
        float: The mean metric score over the input data.
    """
    # Delegate to the implementation in library.config to avoid code duplication
    return laplace_log_likelihood(y_true, y_pred, sigma)
