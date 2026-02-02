import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

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
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metric_score(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): Ground truth FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        sigma (np.array or torch.Tensor): Predicted confidence (standard deviation).

    Returns:
        float: The mean metric score across all samples.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma = np.array(sigma)

    # 1. Clip sigma (Confidence)
    # sigma_clipped = max(sigma, 70)
    sigma_clipped = np.maximum(sigma, Config.SIGMA_CLIP)

    # 2. Calculate Delta (Absolute Error)
    # delta = min(|FVC_true - FVC_predicted|, 1000)
    abs_error = np.abs(y_true - y_pred)
    delta = np.minimum(abs_error, Config.ERROR_CLIP)

    # 3. Compute Metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the average score
    return np.mean(metric)
