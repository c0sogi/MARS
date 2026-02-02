import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Ensures deterministic behavior for CUDA operations.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training epochs.
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


def score_function(y_true, y_pred, sigma):
    """
    Calculates the competition metric: Modified Laplace Log Likelihood.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (numpy array or torch tensor).
        y_pred: Predicted FVC values (numpy array or torch tensor).
        sigma: Predicted confidence/std dev (numpy array or torch tensor).

    Returns:
        float: The average metric score (negative value, higher is better).
    """
    # Convert tensors to numpy if necessary for calculation
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Retrieve constants from Config
    min_sigma = Config.MIN_UNCERTAINTY
    max_error = Config.MAX_ERROR

    # 1. Clip uncertainty (sigma)
    sigma_clipped = np.maximum(sigma, min_sigma)

    # 2. Calculate absolute error and clip it
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, max_error)

    # 3. Compute metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the average score over the batch/set
    return np.mean(metric)
