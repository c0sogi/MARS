import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
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


def get_score(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (Tensor or ndarray).
        y_pred: Predicted FVC values (Tensor or ndarray).
        sigma: Predicted Confidence/Std Dev (Tensor or ndarray).

    Returns:
        float: The mean metric score.
    """
    # Convert numpy arrays to torch tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(sigma, np.ndarray):
        sigma = torch.tensor(sigma)

    # Ensure inputs are float tensors
    y_true = y_true.float()
    y_pred = y_pred.float()
    sigma = sigma.float()

    # align devices
    device = y_true.device
    if y_pred.device != device:
        y_pred = y_pred.to(device)
    if sigma.device != device:
        sigma = sigma.to(device)

    # Retrieve constants from Config
    min_confidence = Config.MIN_CONFIDENCE
    max_error = Config.MAX_ERROR_THRESHOLD

    # Apply clipping
    sigma_clipped = torch.clamp(sigma, min=min_confidence)

    abs_diff = torch.abs(y_true - y_pred)
    delta = torch.clamp(abs_diff, max=max_error)

    # Calculate metric
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))
    metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

    return torch.mean(metric).item()
