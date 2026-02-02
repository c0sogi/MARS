import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
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
    Used for tracking losses and metrics during training.
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


def metric_laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor): Ground truth FVC values.
        y_pred (torch.Tensor): Predicted FVC values.
        sigma (torch.Tensor): Predicted Confidence (standard deviation).

    Returns:
        torch.Tensor: The mean metric value for the batch.
    """
    # Ensure inputs are tensors (handling potential numpy inputs if used in inference)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, dtype=torch.float32)

    # Move to same device if necessary
    if y_true.device != y_pred.device:
        y_true = y_true.to(y_pred.device)
    if sigma.device != y_pred.device:
        sigma = sigma.to(y_pred.device)

    # 1. Clip Confidence (sigma)
    # sigma_clipped = max(sigma, 70)
    sigma_clipped = torch.clamp(sigma, min=Config.MIN_CONFIDENCE)

    # 2. Calculate Absolute Error (Delta)
    absolute_error = torch.abs(y_true - y_pred)

    # 3. Clip Error
    # delta = min(|FVC_true - FVC_pred|, 1000)
    delta = torch.clamp(absolute_error, max=Config.MAX_ERROR_THRESHOLD)

    # 4. Compute Metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=y_pred.device))

    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = torch.log(sqrt_2 * sigma_clipped)

    metric = -term1 - term2

    # Return the mean over the batch
    return torch.mean(metric)
