import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
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

    Args:
        y_true: Array-like or Tensor of true FVC values.
        y_pred: Array-like or Tensor of predicted FVC values.
        sigma: Array-like or Tensor of predicted confidence (standard deviation).

    Returns:
        float: The average metric score (higher is better, values are negative).
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float arrays
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    sigma = np.array(sigma, dtype=np.float32)

    # Clip sigma (confidence) at 70 ml
    sigma_clipped = np.maximum(sigma, Config.MIN_CONFIDENCE_CLIP)

    # Calculate absolute error and clip at 1000 ml
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, Config.MAX_ERROR_CLIP)

    # Calculate metric formula: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    # Return the mean metric across the batch
    return np.mean(metric)
