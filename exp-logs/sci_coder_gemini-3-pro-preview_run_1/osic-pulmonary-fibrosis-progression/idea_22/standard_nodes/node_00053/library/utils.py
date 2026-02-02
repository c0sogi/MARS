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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (array-like): Ground truth FVC values.
        y_pred (array-like): Predicted FVC values.
        sigma (array-like): Predicted confidence (standard deviation) values.

    Returns:
        float: The average metric score across all samples.
    """
    # Convert inputs to numpy arrays for vectorized operations
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sigma = np.asarray(sigma)

    # Clip confidence values (sigma) at 70 ml
    sigma_clipped = np.maximum(sigma, 70)

    # Calculate absolute error and clip at 1000 ml
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, 1000)

    # Calculate the metric
    sq2 = np.sqrt(2)
    metric = -(sq2 * delta) / sigma_clipped - np.log(sq2 * sigma_clipped)

    # Return the average score
    return np.mean(metric)
