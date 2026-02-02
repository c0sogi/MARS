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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the competition.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (torch.Tensor): Ground truth FVC values.
        y_pred (torch.Tensor): Predicted FVC values.
        sigma (torch.Tensor): Predicted confidence (standard deviation).

    Returns:
        torch.Tensor: The mean metric score for the batch (scalar).
    """
    # Ensure inputs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(sigma)

    # Move to same device if necessary (assuming y_pred is on the correct device)
    device = y_pred.device
    y_true = y_true.to(device)
    sigma = sigma.to(device)

    # 1. Clip sigma (confidence)
    # sigma_clipped = max(sigma, 70)
    sigma_clipped = torch.clamp(sigma, min=Config.SIGMA_CLIP_MIN)

    # 2. Calculate absolute error and clip it
    # delta = min(|FVC_true - FVC_pred|, 1000)
    abs_error = torch.abs(y_true - y_pred)
    delta = torch.clamp(abs_error, max=Config.MAX_ERROR_CLIP)

    # 3. Calculate Metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))

    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = torch.log(sqrt_2 * sigma_clipped)

    metric = -term1 - term2

    return torch.mean(metric)


def smooth_nll_loss(y_true, y_pred, sigma):
    """
    Calculates a smooth Negative Log Likelihood loss aligned with the metric.
    Loss = sqrt(2) * |y_true - y_pred| / sigma + ln(sigma)
    Cite {solution_lesson_node_00138}
    Cite {solution_lesson_node_00066}
    """
    # Ensure sigma is positive and avoid division by zero
    sigma = torch.clamp(sigma, min=1e-6)

    abs_error = torch.abs(y_true - y_pred)
    sqrt_2 = 1.41421356

    # We ignore the constant ln(sqrt(2)) in the loss optimization
    nll = (sqrt_2 * abs_error) / sigma + torch.log(sigma)

    return torch.mean(nll)
