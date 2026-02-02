import os
import random
import numpy as np
import torch


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


def calculate_metric(y_true, y_pred, sigma):
    """
    Calculates the Modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor or np.array): Ground truth FVC values.
        y_pred (torch.Tensor or np.array): Predicted FVC values.
        sigma (torch.Tensor or np.array): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score (negative value, higher is better).
    """
    # Convert numpy arrays to torch tensors if necessary
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma, dtype=torch.float32)

    # Move tensors to the same device (CPU) for calculation to avoid device mismatches
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    sigma = sigma.cpu()

    # Apply clipping as per metric definition
    sigma_clipped = torch.clamp(sigma, min=70)
    delta = torch.clamp(torch.abs(y_true - y_pred), max=1000)

    # Calculate metric
    # Note: np.sqrt(2) is a float, torch handles the broadcasting
    metric = -(np.sqrt(2) * delta) / sigma_clipped - torch.log(
        np.sqrt(2) * sigma_clipped
    )

    # Return the mean over the batch
    return torch.mean(metric).item()
