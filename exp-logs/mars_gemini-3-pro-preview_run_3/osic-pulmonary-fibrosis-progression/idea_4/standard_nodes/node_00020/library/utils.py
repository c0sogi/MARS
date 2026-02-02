import numpy as np
import torch
from library.config import Config, seed_everything


def unscale_data(fvc_pred, sigma_pred):
    """
    Inverse transforms the standardized predictions back to the original scale (ml).

    Logic:
        FVC_final = FVC_pred * std + mean
        Sigma_final = Sigma_pred * std

    Args:
        fvc_pred (np.array or torch.Tensor): Standardized FVC predictions.
        sigma_pred (np.array or torch.Tensor): Scaled confidence predictions.

    Returns:
        tuple: (fvc_unscaled, sigma_unscaled) of the same type as input.
    """
    target_std = Config.TARGET_STD
    target_mean = Config.TARGET_MEAN

    if isinstance(fvc_pred, torch.Tensor):
        fvc_unscaled = fvc_pred * target_std + target_mean
        sigma_unscaled = sigma_pred * target_std
    else:
        fvc_unscaled = fvc_pred * target_std + target_mean
        sigma_unscaled = sigma_pred * target_std

    return fvc_unscaled, sigma_unscaled


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the competition.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or float): True FVC values (in ml).
        y_pred (np.array or float): Predicted FVC values (in ml).
        sigma (np.array or float): Predicted confidence (sigma) values (in ml).

    Returns:
        float: The average metric score across the input batch.
    """
    # Ensure inputs are numpy arrays for calculation
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Handle scalar inputs by converting to 1D arrays
    y_true = np.atleast_1d(y_true)
    y_pred = np.atleast_1d(y_pred)
    sigma = np.atleast_1d(sigma)

    # 1. Clip confidence values
    sigma_clipped = np.maximum(sigma, Config.SIGMA_CLIP)

    # 2. Calculate absolute error and clip it
    abs_diff = np.abs(y_true - y_pred)
    delta = np.minimum(abs_diff, Config.MAX_ERROR)

    # 3. Compute Metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)
