import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood_metric(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric defined for the competition.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (original scale, ml). Can be np.array or torch.Tensor.
        y_pred: Predicted FVC values (original scale, ml). Can be np.array or torch.Tensor.
        sigma: Predicted confidence/std dev (original scale, ml). Can be np.array or torch.Tensor.

    Returns:
        float: The average metric score across the batch.
    """
    # Convert to numpy if inputs are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # constants
    sqrt_2 = np.sqrt(2)

    # Clipping sigma
    sigma_clipped = np.maximum(sigma, 70)

    # Calculating delta with threshold
    abs_diff = np.abs(y_true - y_pred)
    delta = np.minimum(abs_diff, 1000)

    # Calculating the metric components
    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = np.log(sqrt_2 * sigma_clipped)

    metric = -term1 - term2

    return np.mean(metric)


def inverse_scale_predictions(pred_mean_scaled, pred_sigma_scaled):
    """
    Transforms model predictions from the standardized Z-score scale back to the original ml scale.

    Args:
        pred_mean_scaled: Predicted mean in standardized scale.
        pred_sigma_scaled: Predicted standard deviation in standardized scale.

    Returns:
        tuple: (pred_mean_original, pred_sigma_original)
    """
    target_std = Config.TARGET_STD
    target_mean = Config.TARGET_MEAN

    # Inverse scale the mean: x = z * std + mean
    pred_mean_original = pred_mean_scaled * target_std + target_mean

    # Inverse scale the sigma: sigma_orig = sigma_scaled * std
    # Note: Sigma is a measure of spread, so it is only multiplied by the scale factor, not shifted.
    pred_sigma_original = pred_sigma_scaled * target_std

    return pred_mean_original, pred_sigma_original
