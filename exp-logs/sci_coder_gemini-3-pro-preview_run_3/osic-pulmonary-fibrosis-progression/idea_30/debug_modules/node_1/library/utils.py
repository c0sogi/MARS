import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred, sigma_pred):
    """
    Computes the modified Laplace Log Likelihood metric defined in the task.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth FVC values.
        y_pred (np.ndarray or torch.Tensor): Predicted FVC values.
        sigma_pred (np.ndarray or torch.Tensor): Predicted confidence (sigma) values.

    Returns:
        float: The average metric score across the batch.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma_pred, torch.Tensor):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    # Ensure inputs are float64 for precision
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma_pred = np.array(sigma_pred, dtype=np.float64)

    # 1. Clip the confidence values (sigma) at 70 ml
    sigma_clipped = np.maximum(sigma_pred, 70)

    # 2. Calculate absolute error (delta) and threshold at 1000 ml
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, 1000)

    # 3. Compute the metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the average score
    return np.mean(metric)
