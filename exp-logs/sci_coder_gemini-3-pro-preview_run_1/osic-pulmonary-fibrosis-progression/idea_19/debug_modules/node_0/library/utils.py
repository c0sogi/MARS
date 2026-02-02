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

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic algorithms can be slower but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric used in the competition.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: Ground truth FVC values (array-like or Tensor).
        y_pred: Predicted FVC values (array-like or Tensor).
        sigma: Predicted Confidence (standard deviation) values (array-like or Tensor).

    Returns:
        float: The average metric score (higher is better, values are negative).
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure inputs are float arrays
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    sigma = np.array(sigma, dtype=np.float32)

    # 1. Clip the confidence (sigma)
    # The metric clips confidence at 70 ml to reflect approximate measurement uncertainty
    sigma_clipped = np.maximum(sigma, Config.METRIC_MIN_CONF)

    # 2. Calculate the absolute error (delta)
    # The metric thresholds error at 1000 ml to avoid large errors excessively penalizing results
    abs_error = np.abs(y_true - y_pred)
    delta = np.minimum(abs_error, Config.METRIC_CLIP_ERR)

    # 3. Compute the metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the average score
    return np.mean(metric)
