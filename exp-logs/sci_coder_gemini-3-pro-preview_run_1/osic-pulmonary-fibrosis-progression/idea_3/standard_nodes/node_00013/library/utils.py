import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
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


def laplace_log_likelihood(true_fvc, pred_fvc, pred_sigma, clamp_delta=True):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        Metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Constraints:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)

    Args:
        true_fvc: True FVC values (Tensor or numpy array).
        pred_fvc: Predicted FVC values (Tensor or numpy array).
        pred_sigma: Predicted Confidence (sigma) values (Tensor or numpy array).
        clamp_delta (bool): Whether to clip the error at 1000 ml. Defaults to True.

    Returns:
        torch.Tensor: The mean metric value (scalar).
    """
    # Convert numpy arrays to tensors if necessary
    if isinstance(true_fvc, np.ndarray):
        true_fvc = torch.from_numpy(true_fvc).float()
    if isinstance(pred_fvc, np.ndarray):
        pred_fvc = torch.from_numpy(pred_fvc).float()
    if isinstance(pred_sigma, np.ndarray):
        pred_sigma = torch.from_numpy(pred_sigma).float()

    # Ensure all tensors are on the same device
    if isinstance(true_fvc, torch.Tensor):
        device = true_fvc.device
        if isinstance(pred_fvc, torch.Tensor):
            pred_fvc = pred_fvc.to(device)
        if isinstance(pred_sigma, torch.Tensor):
            pred_sigma = pred_sigma.to(device)

    # Constants from Config
    sigma_clip_val = Config.MIN_CONFIDENCE
    delta_clip_val = Config.MAX_ERROR
    sqrt_2 = np.sqrt(2)

    # Clip confidence (sigma)
    # The metric requires confidence to be at least 70 ml
    sigma_clipped = torch.clamp(pred_sigma, min=sigma_clip_val)

    # Calculate absolute error (delta)
    abs_error = torch.abs(true_fvc - pred_fvc)

    # Clip error at 1000 ml
    if clamp_delta:
        delta = torch.clamp(abs_error, max=delta_clip_val)
    else:
        delta = abs_error

    # Calculate metric terms
    # Term 1: - (sqrt(2) * delta) / sigma_clipped
    term1 = -(sqrt_2 * delta) / sigma_clipped

    # Term 2: - ln(sqrt(2) * sigma_clipped)
    term2 = -torch.log(sqrt_2 * sigma_clipped)

    # Combine terms
    metric = term1 + term2

    # Return the mean metric over the batch
    return torch.mean(metric)
