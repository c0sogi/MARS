import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic algorithms ensure reproducibility but might be slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inverse_transform(mu_scaled, sigma_scaled):
    """
    Converts standardized model predictions back to the original ml scale.

    Uses the global target mean and std from Config to reverse the z-score normalization.

    Args:
        mu_scaled (torch.Tensor or np.ndarray): Predicted FVC (standardized).
        sigma_scaled (torch.Tensor or np.ndarray): Predicted uncertainty (standardized).

    Returns:
        tuple: (mu_original, sigma_original) in milliliters (ml).
    """
    # Convert to numpy if input is a torch Tensor
    if isinstance(mu_scaled, torch.Tensor):
        mu_scaled = mu_scaled.detach().cpu().numpy()
    if isinstance(sigma_scaled, torch.Tensor):
        sigma_scaled = sigma_scaled.detach().cpu().numpy()

    # Reverse standardization: x = z * std + mean
    mu_original = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN

    # Scale sigma: sigma_orig = sigma_scaled * std
    sigma_original = sigma_scaled * Config.TARGET_STD

    return mu_original, sigma_original


def calculate_metric(y_true, y_pred, sigma_pred):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth FVC values (ml).
        y_pred (np.ndarray or torch.Tensor): Predicted FVC values (ml).
        sigma_pred (np.ndarray or torch.Tensor): Predicted confidence/std (ml).

    Returns:
        float: The average metric score across the batch.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma_pred, torch.Tensor):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    # 1. Clip sigma at 70 ml
    sigma_clipped = np.maximum(sigma_pred, 70)

    # 2. Calculate absolute error and clip at 1000 ml
    delta = np.minimum(np.abs(y_true - y_pred), 1000)

    # 3. Compute metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)


def print_metrics(phase, epoch, loss, score):
    """
    Prints validation metrics with full precision.

    Args:
        phase (str): 'Train' or 'Val'.
        epoch (int): Current epoch number.
        loss (float): Calculated loss.
        score (float): Calculated metric score.
    """
    print(f"Phase: {phase} | Epoch: {epoch} | Loss: {loss} | Score: {score}")
