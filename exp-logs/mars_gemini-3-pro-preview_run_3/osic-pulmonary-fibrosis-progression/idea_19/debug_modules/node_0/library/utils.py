import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


def score_function(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric defined for the competition.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor or np.ndarray): True FVC values.
        y_pred (torch.Tensor or np.ndarray): Predicted FVC values.
        sigma (torch.Tensor or np.ndarray): Predicted confidence (std dev).

    Returns:
        float: The mean metric score over the batch.
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if isinstance(sigma, np.ndarray):
        sigma = torch.tensor(sigma, dtype=torch.float32)

    # Ensure tensors are on the same device (cpu for metric calculation is usually safer/easier)
    # but if they are already on gpu, we keep them there to avoid transfers
    if y_true.device != y_pred.device:
        y_true = y_true.to(y_pred.device)

    # 1. Clip sigma: max(sigma, 70)
    # Using Config.SIGMA_CLIP which is 70.0
    sigma_clipped = torch.clamp(sigma, min=Config.SIGMA_CLIP)

    # 2. Calculate absolute error (delta)
    absolute_error = torch.abs(y_true - y_pred)

    # 3. Clip delta: min(|true - pred|, 1000)
    # Using Config.MAX_ERROR which is 1000.0
    delta_clipped = torch.clamp(absolute_error, max=Config.MAX_ERROR)

    # 4. Compute Metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    # Config.METRIC_CONSTANT is sqrt(2)

    term1 = -(Config.METRIC_CONSTANT * delta_clipped) / sigma_clipped
    term2 = -torch.log(Config.METRIC_CONSTANT * sigma_clipped)

    metric = term1 + term2

    return torch.mean(metric).item()
