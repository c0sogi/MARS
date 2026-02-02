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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking losses and metrics during training.
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


def compute_metric(true_fvc, pred_fvc, pred_sigma):
    """
    Computes the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        true_fvc (np.array or list): Ground truth FVC values.
        pred_fvc (np.array or list): Predicted FVC values.
        pred_sigma (np.array or list): Predicted confidence (sigma) values.

    Returns:
        float: The average metric score (higher is better, typically negative).
    """
    # Ensure inputs are numpy arrays
    true_fvc = np.array(true_fvc, dtype=np.float64)
    pred_fvc = np.array(pred_fvc, dtype=np.float64)
    pred_sigma = np.array(pred_sigma, dtype=np.float64)

    # Clip sigma to reflect approximate measurement uncertainty
    sigma_clipped = np.maximum(pred_sigma, 70)

    # Calculate absolute error and threshold it
    delta = np.abs(true_fvc - pred_fvc)
    delta = np.minimum(delta, 1000)

    # Compute metric components
    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the average score
    return np.mean(metric_values)
