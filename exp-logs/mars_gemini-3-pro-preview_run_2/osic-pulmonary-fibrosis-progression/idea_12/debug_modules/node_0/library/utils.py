import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def score_function(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (array-like or Tensor)
        y_pred: Predicted FVC values (array-like or Tensor)
        sigma: Predicted confidence/std (array-like or Tensor)

    Returns:
        float: The average metric score (higher is better, typically negative).
    """
    # Handle PyTorch Tensors: detach, move to CPU, convert to numpy
    if hasattr(y_true, "cpu"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.detach().cpu().numpy()
    if hasattr(sigma, "cpu"):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float64 numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Metric Constants
    MIN_SIGMA = 70.0
    MAX_DELTA = 1000.0
    SQRT_2 = np.sqrt(2)

    # Apply Clipping
    sigma_clipped = np.maximum(sigma, MIN_SIGMA)
    delta = np.abs(y_true - y_pred)
    delta_clipped = np.minimum(delta, MAX_DELTA)

    # Calculate Metric
    # Term 1: - (sqrt(2) * delta) / sigma
    term1 = (SQRT_2 * delta_clipped) / sigma_clipped
    # Term 2: - ln(sqrt(2) * sigma)
    term2 = np.log(SQRT_2 * sigma_clipped)

    metric = -term1 - term2

    return np.mean(metric)


def save_results(df: pd.DataFrame, path: str):
    """
    Safely saves a pandas DataFrame to a CSV file.
    Creates the parent directory if it does not exist.

    Args:
        df: The DataFrame to save.
        path: The destination file path.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    df.to_csv(path, index=False)


def log_metrics(metrics: dict, prefix: str = ""):
    """
    Prints a dictionary of metrics to stdout without rounding.

    Args:
        metrics: Dictionary where keys are metric names and values are scores.
        prefix: Optional string to prepend to the log message.
    """
    parts = [f"{k}: {v}" for k, v in metrics.items()]
    message = " | ".join(parts)

    if prefix:
        print(f"{prefix}: {message}")
    else:
        print(message)
