import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric_score(y_true, y_pred_mean, y_pred_sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): True FVC values.
        y_pred_mean (np.array or torch.Tensor): Predicted FVC values.
        y_pred_sigma (np.array or torch.Tensor): Predicted uncertainty (sigma).

    Returns:
        float: The average metric score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_mean, torch.Tensor):
        y_pred_mean = y_pred_mean.detach().cpu().numpy()
    if isinstance(y_pred_sigma, torch.Tensor):
        y_pred_sigma = y_pred_sigma.detach().cpu().numpy()

    # Ensure inputs are flattened
    y_true = y_true.flatten()
    y_pred_mean = y_pred_mean.flatten()
    y_pred_sigma = y_pred_sigma.flatten()

    # Constants
    sigma_clip_val = Config.MIN_SIGMA
    delta_clip_val = Config.MAX_ERROR
    sqrt_2 = np.sqrt(2)

    # Clipping sigma
    sigma_clipped = np.maximum(y_pred_sigma, sigma_clip_val)

    # Calculating delta and clipping
    delta = np.abs(y_true - y_pred_mean)
    delta_clipped = np.minimum(delta, delta_clip_val)

    # Metric calculation
    # metric = - (sqrt(2) * delta_clipped / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    term1 = (sqrt_2 * delta_clipped) / sigma_clipped
    term2 = np.log(sqrt_2 * sigma_clipped)
    score = -term1 - term2

    return np.mean(score)
