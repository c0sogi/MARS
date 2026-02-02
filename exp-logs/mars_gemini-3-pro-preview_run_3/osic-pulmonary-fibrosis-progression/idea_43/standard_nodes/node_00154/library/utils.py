import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred, sigma_pred):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the competition.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: True FVC values. Can be numpy array or torch.Tensor.
        y_pred: Predicted FVC values. Can be numpy array or torch.Tensor.
        sigma_pred: Predicted Confidence (sigma). Can be numpy array or torch.Tensor.

    Returns:
        float: The average metric score over the input data.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma_pred, torch.Tensor):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    # Ensure data types are float for calculation
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sigma_pred = np.asarray(sigma_pred, dtype=np.float64)

    # Apply clipping to sigma (confidence)
    # The confidence values are clipped at 70 ml to reflect approximate measurement uncertainty
    sigma_clipped = np.maximum(sigma_pred, Config.MIN_CONFIDENCE)

    # Calculate absolute error
    delta = np.abs(y_true - y_pred)

    # Threshold the error at 1000 ml to avoid large errors adversely penalizing results
    delta = np.minimum(delta, Config.MAX_ERROR)

    # Calculate the metric
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    metric = -(Config.SQRT_2 * delta) / sigma_clipped - np.log(
        Config.SQRT_2 * sigma_clipped
    )

    # Return the average metric
    return np.mean(metric)
