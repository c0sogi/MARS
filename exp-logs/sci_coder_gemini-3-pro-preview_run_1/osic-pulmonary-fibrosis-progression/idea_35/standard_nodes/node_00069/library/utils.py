import os
import random
import numpy as np
import torch


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred, y_conf):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (numpy array or torch tensor).
        y_pred: Predicted FVC values (numpy array or torch tensor).
        y_conf: Predicted Confidence (sigma) values (numpy array or torch tensor).

    Returns:
        float: The average metric score across all samples.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_conf, torch.Tensor):
        y_conf = y_conf.detach().cpu().numpy()

    # Ensure inputs are float64 for precision
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    y_conf = np.array(y_conf, dtype=np.float64)

    # 1. Clip confidence values (sigma) to a minimum of 70
    sigma_clipped = np.maximum(y_conf, 70)

    # 2. Calculate absolute error (delta) and clip it to a maximum of 1000
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, 1000)

    # 3. Calculate the metric
    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the mean score
    return np.mean(metric_values)
