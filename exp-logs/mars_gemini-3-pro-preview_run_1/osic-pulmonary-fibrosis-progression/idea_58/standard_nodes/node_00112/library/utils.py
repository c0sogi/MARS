import os
import sys
import random
import numpy as np
import torch
from library.config import Config


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic settings for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
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


class Logger:
    """
    Logs messages to both a file and the standard output.
    """

    def __init__(self, log_file):
        self.log_file = log_file
        if self.log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            # Initialize/Clear the file
            with open(self.log_file, "w") as f:
                pass

    def log(self, message):
        """
        Prints message to stdout and appends to log file.
        """
        print(message)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (Tensor or numpy array).
        y_pred: Predicted FVC values (Tensor or numpy array).
        sigma: Predicted Confidence/Std Dev (Tensor or numpy array).

    Returns:
        torch.Tensor: The mean metric score.
    """
    # Convert numpy arrays to tensors if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma)

    # Ensure consistent device and type
    device = y_true.device
    y_true = y_true.float()
    y_pred = y_pred.to(device).float()
    sigma = sigma.to(device).float()

    # Apply clipping constraints from Config
    sigma_clipped = torch.clamp(sigma, min=Config.SIGMA_MIN)

    abs_diff = torch.abs(y_true - y_pred)
    delta = torch.clamp(abs_diff, max=Config.ERROR_MAX)

    # Calculate metric
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))
    metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

    return torch.mean(metric)
