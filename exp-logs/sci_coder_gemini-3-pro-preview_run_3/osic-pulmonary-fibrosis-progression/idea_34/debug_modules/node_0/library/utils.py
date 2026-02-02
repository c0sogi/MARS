import os
import random
import numpy as np
import torch
import math
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
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
    Computes the modified Laplace Log Likelihood metric for the competition.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor or np.ndarray): True FVC values.
        y_pred (torch.Tensor or np.ndarray): Predicted FVC values.
        sigma (torch.Tensor or np.ndarray): Predicted confidence (std dev).

    Returns:
        float: The mean metric score over the batch.
    """
    # Convert to tensors if numpy arrays are passed
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, dtype=torch.float32)

    # Ensure all tensors are on the same device
    device = y_pred.device
    y_true = y_true.to(device)
    sigma = sigma.to(device)

    # Constants from Config
    sigma_clip_val = Config.QUANTILE_CLIP
    max_error_val = Config.MAX_ERROR

    # Calculate metric components
    sigma_clipped = torch.clamp(sigma, min=sigma_clip_val)
    delta = torch.abs(y_true - y_pred)
    delta = torch.clamp(delta, max=max_error_val)

    sq2 = math.sqrt(2)

    # Calculate score
    metric = -(sq2 * delta) / sigma_clipped - torch.log(sq2 * sigma_clipped)

    # Return mean score
    return torch.mean(metric).item()
