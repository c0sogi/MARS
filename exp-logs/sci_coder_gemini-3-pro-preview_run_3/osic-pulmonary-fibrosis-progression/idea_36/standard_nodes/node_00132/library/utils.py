import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric defined in the task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: True FVC values (numpy array, list, or torch.Tensor).
        y_pred: Predicted FVC values (numpy array, list, or torch.Tensor).
        sigma: Predicted confidence/std values (numpy array, list, or torch.Tensor).

    Returns:
        float: The mean metric score across the input batch.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    sigma = np.array(sigma, dtype=np.float32)

    # Apply clipping constraints defined in the metric
    sigma_clipped = np.maximum(sigma, Config.METRIC_CLIP_SIGMA)
    delta = np.minimum(np.abs(y_true - y_pred), Config.METRIC_MAX_ERROR)

    # Calculate metric
    # Note: The metric is negative, and higher (less negative) is better.
    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)
