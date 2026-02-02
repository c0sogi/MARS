import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric as defined in the task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth FVC values.
        y_pred (torch.Tensor or np.ndarray): Predicted FVC values.
        sigma (torch.Tensor or np.ndarray): Predicted confidence (std dev).

    Returns:
        float: The average metric score (higher is better, values are negative).
    """
    # Convert tensors to numpy arrays if necessary
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(sigma):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float for calculation
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    sigma = sigma.astype(np.float64)

    # Apply clipping constraints
    sigma_clipped = np.maximum(sigma, 70)
    delta = np.minimum(np.abs(y_true - y_pred), 1000)

    # Calculate metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)


class MetricMonitor:
    """
    A utility class to compute and store the average and current value of a metric.
    Useful for tracking loss and scores during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
