import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred, y_std):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or list): True FVC measurements.
        y_pred (np.array or list): Predicted FVC measurements.
        y_std (np.array or list): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score across all samples.
    """
    # Convert inputs to numpy arrays with high precision
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    y_std = np.array(y_std, dtype=np.float64)

    # Constants from Config
    sigma_clip_val = Config.CONFIDENCE_CLIP
    delta_clip_val = Config.MAX_ERROR_CLIP
    sqrt_2 = Config.SQRT2

    # Apply clipping logic
    # sigma_clipped = max(sigma, 70)
    sigma_clipped = np.maximum(y_std, sigma_clip_val)

    # delta = min(|FVC_true - FVC_pred|, 1000)
    delta = np.abs(y_true - y_pred)
    delta = np.minimum(delta, delta_clip_val)

    # Calculate metric
    # metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    term_1 = (sqrt_2 * delta) / sigma_clipped
    term_2 = np.log(sqrt_2 * sigma_clipped)

    metric = -term_1 - term_2

    # Return the average score
    return np.mean(metric)
