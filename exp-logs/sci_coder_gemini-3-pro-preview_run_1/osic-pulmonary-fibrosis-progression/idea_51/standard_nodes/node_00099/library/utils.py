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
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric as defined in the task.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth FVC values.
        y_pred (torch.Tensor or np.ndarray): Predicted FVC values.
        sigma (torch.Tensor or np.ndarray): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score across the input batch.
               Values are negative; higher (closer to 0) is better.
    """
    # Detach and move to CPU if inputs are PyTorch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are numpy arrays (handles lists or other iterables)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sigma = np.array(sigma)

    # 1. Clip confidence values (sigma) at 70 ml
    sigma_clipped = np.maximum(sigma, 70)

    # 2. Calculate absolute error and clip at 1000 ml
    absolute_error = np.abs(y_true - y_pred)
    delta = np.minimum(absolute_error, 1000)

    # 3. Compute the metric formula
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the mean metric over the batch
    return np.mean(metric_values)
