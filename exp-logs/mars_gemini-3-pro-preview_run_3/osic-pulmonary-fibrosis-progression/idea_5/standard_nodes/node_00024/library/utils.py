import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood_loss(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood loss for training.
    Formula: L = |y_true - y_pred| / sigma + log(sigma)

    Args:
        y_true (torch.Tensor): Ground truth FVC values (typically standardized).
        y_pred (torch.Tensor): Predicted FVC values (typically standardized).
        sigma (torch.Tensor): Predicted confidence (standard deviation), must be positive.

    Returns:
        torch.Tensor: Scalar loss value (mean over batch).
    """
    # Calculate absolute error
    delta = torch.abs(y_true - y_pred)

    # Calculate loss: delta/sigma + log(sigma)
    # Note: The constant sqrt(2) is omitted for training stability/simplification
    # as per the provided Idea description.
    loss = (delta / sigma) + torch.log(sigma)

    return torch.mean(loss)


def score_metric(y_true, y_pred, sigma):
    """
    Calculates the competition evaluation metric.

    Args:
        y_true (np.ndarray): Ground truth FVC values in ml.
        y_pred (np.ndarray): Predicted FVC values in ml.
        sigma (np.ndarray): Predicted confidence in ml.

    Returns:
        float: The mean metric score.
    """
    # Clip sigma to reflect approximate measurement uncertainty
    sigma_clipped = np.maximum(sigma, Config.MIN_CONFIDENCE)

    # Calculate absolute error and clip it to avoid large errors adversely penalizing results
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, Config.MAX_ERROR)

    # Calculate metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)


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
