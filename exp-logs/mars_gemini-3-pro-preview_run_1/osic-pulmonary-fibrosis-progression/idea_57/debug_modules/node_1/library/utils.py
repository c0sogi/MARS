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


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric for the Pulmonary Fibrosis competition.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): Ground truth FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        sigma (np.array or torch.Tensor): Predicted confidence (std dev).

    Returns:
        float: The average metric score across the batch.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Clip confidence values at 70ml
    sigma_clipped = np.maximum(sigma, Config.CONFIDENCE_CLIP)

    # Calculate absolute error
    delta = np.abs(y_true - y_pred)

    # Clip error at 1000ml to avoid large penalties for outliers
    delta = np.minimum(delta, Config.MAX_ERROR)

    # Compute the metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    # Return the mean score
    return np.mean(metric)
