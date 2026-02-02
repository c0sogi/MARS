import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth FVC values.
        y_pred (torch.Tensor or np.ndarray): Predicted FVC values.
        sigma (torch.Tensor or np.ndarray): Predicted confidence (standard deviation).

    Returns:
        float: The average metric value over the batch.
    """
    # Convert tensors to numpy arrays for calculation if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Clip confidence values (sigma)
    # The metric requires sigma to be at least 70 ml
    sigma_clipped = np.maximum(sigma, Config.MIN_CONFIDENCE)

    # Calculate absolute error (delta)
    absolute_error = np.abs(y_true - y_pred)

    # Clip error at 1000 ml
    delta = np.minimum(absolute_error, Config.MAX_ERROR)

    # Calculate metric
    # Note: Metric values will be negative, higher is better
    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)
