import os
import random
import numpy as np
import torch
import math


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


def score_function(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor): Ground truth FVC values.
        y_pred (torch.Tensor): Predicted FVC values.
        sigma (torch.Tensor): Predicted confidence (standard deviation).

    Returns:
        torch.Tensor: The mean metric score for the batch.
    """
    # Ensure inputs are on the same device
    if y_true.device != y_pred.device:
        y_true = y_true.to(y_pred.device)
    if sigma.device != y_pred.device:
        sigma = sigma.to(y_pred.device)

    # Clip sigma to a minimum of 70
    sigma_clipped = torch.clamp(sigma, min=70)

    # Calculate absolute error
    delta = torch.abs(y_true - y_pred)

    # Clip delta to a maximum of 1000
    delta = torch.clamp(delta, max=1000)

    # Calculate metric components
    sqrt_2 = math.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

    # Return the average score
    return torch.mean(metric)


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
