import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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


def inverse_transform(pred_fvc_scaled, pred_sigma_scaled):
    """
    Converts Z-scored model predictions back to the original ml scale.

    Args:
        pred_fvc_scaled (np.array or torch.Tensor): The predicted FVC in Z-score space.
        pred_sigma_scaled (np.array or torch.Tensor): The predicted uncertainty (sigma) in Z-score space.

    Returns:
        tuple: (fvc_ml, sigma_ml) in milliliters.
    """
    # Inverse Z-score for the mean: x = z * std + mean
    fvc_ml = pred_fvc_scaled * Config.TARGET_STD + Config.TARGET_MEAN

    # Inverse scaling for standard deviation: sigma_x = sigma_z * std
    # Note: We do not add the mean to the standard deviation
    sigma_ml = pred_sigma_scaled * Config.TARGET_STD

    return fvc_ml, sigma_ml


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric defined for the task.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array): Ground truth FVC values (in ml).
        y_pred (np.array): Predicted FVC values (in ml).
        sigma (np.array): Predicted confidence/uncertainty (in ml).

    Returns:
        float: The average metric score (negative, higher is better).
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # 1. Clip sigma at 70 ml
    sigma_clipped = np.maximum(sigma, 70)

    # 2. Calculate absolute error and clip at 1000 ml
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, 1000)

    # 3. Compute metric components
    # Using Config.SQRT_2 for consistency if needed, but defining locally for independence
    sqrt_2 = np.sqrt(2)

    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = np.log(sqrt_2 * sigma_clipped)

    # Final metric calculation
    metric = -term1 - term2

    return np.mean(metric)
