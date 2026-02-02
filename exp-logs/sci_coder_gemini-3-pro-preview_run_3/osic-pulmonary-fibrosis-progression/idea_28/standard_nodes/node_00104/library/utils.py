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

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred, sigma_pred):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    Metric Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (numpy array or torch.Tensor).
        y_pred: Predicted FVC values (numpy array or torch.Tensor).
        sigma_pred: Predicted confidence/sigma values (numpy array or torch.Tensor).

    Returns:
        float: The average metric score across the batch/dataset.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma_pred, torch.Tensor):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    # Apply clipping constraints defined in the metric
    sigma_clipped = np.maximum(sigma_pred, Config.CONFIDENCE_CLIP)

    # Calculate absolute error and clip at 1000 ml
    abs_error = np.abs(y_true - y_pred)
    delta = np.minimum(abs_error, Config.MAX_ERROR_CLIP)

    # Compute the metric
    # Note: np.log is the natural logarithm (ln)
    metric = -(Config.SQRT_2 * delta) / sigma_clipped - np.log(
        Config.SQRT_2 * sigma_clipped
    )

    # Return the mean score
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
