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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
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
    Computes the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor): True FVC values.
        y_pred (torch.Tensor): Predicted FVC values.
        sigma (torch.Tensor): Predicted confidence (standard deviation).

    Returns:
        torch.Tensor: The mean metric score for the batch.
    """
    # Ensure inputs are on the same device and are floats
    y_true = y_true.float()
    y_pred = y_pred.float()
    sigma = sigma.float()

    # Constants
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=y_true.device))

    # Calculate absolute error
    absolute_error = torch.abs(y_true - y_pred)

    # Apply clipping rules defined in the metric
    # Delta is clipped at 1000 ml
    delta = torch.clamp(absolute_error, max=Config.METRIC_MAX_ERR)

    # Sigma is clipped at 70 ml
    sigma_clipped = torch.clamp(sigma, min=Config.METRIC_MIN_CONF)

    # Calculate metric
    # term1 = (sqrt(2) * delta) / sigma_clipped
    term1 = (sqrt_2 * delta) / sigma_clipped

    # term2 = ln(sqrt(2) * sigma_clipped)
    term2 = torch.log(sqrt_2 * sigma_clipped)

    # metric = - term1 - term2
    metric = -term1 - term2

    # Return the average score
    return torch.mean(metric)
