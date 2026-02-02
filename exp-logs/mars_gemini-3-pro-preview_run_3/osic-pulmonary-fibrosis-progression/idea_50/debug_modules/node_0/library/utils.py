import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior for cuDNN
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


def calculate_metric(y_true, y_pred, y_std):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): True FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        y_std (np.array or torch.Tensor): Predicted Confidence (sigma) values.

    Returns:
        float: The average metric score over the batch.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_std, torch.Tensor):
        y_std = y_std.detach().cpu().numpy()

    # Ensure inputs are flat arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_std = y_std.flatten()

    # Constants
    sigma_clip_val = Config.METRIC_CLIP_SIGMA
    delta_clip_val = 1000.0
    sqrt_2 = np.sqrt(2)

    # 1. Clip Confidence (Sigma)
    # sigma_clipped = max(sigma, 70)
    sigma_clipped = np.maximum(y_std, sigma_clip_val)

    # 2. Calculate Delta with clipping
    # delta = min(|true - pred|, 1000)
    abs_diff = np.abs(y_true - y_pred)
    delta = np.minimum(abs_diff, delta_clip_val)

    # 3. Calculate Metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = np.log(sqrt_2 * sigma_clipped)

    metric = -term1 - term2

    return np.mean(metric)
