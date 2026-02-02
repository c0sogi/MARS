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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metric(true_fvc, pred_fvc, pred_sigma):
    """
    Computes the modified Laplace Log Likelihood metric as defined in the competition task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        true_fvc (array-like): Ground truth FVC values.
        pred_fvc (array-like): Predicted FVC values.
        pred_sigma (array-like): Predicted confidence (sigma) values.

    Returns:
        float: The average metric score across the input batch.
    """
    # Convert inputs to numpy arrays for consistent calculation
    # Detach from graph if they are torch tensors
    if isinstance(true_fvc, torch.Tensor):
        true_fvc = true_fvc.detach().cpu().numpy()
    if isinstance(pred_fvc, torch.Tensor):
        pred_fvc = pred_fvc.detach().cpu().numpy()
    if isinstance(pred_sigma, torch.Tensor):
        pred_sigma = pred_sigma.detach().cpu().numpy()

    true_fvc = np.array(true_fvc, dtype=np.float64)
    pred_fvc = np.array(pred_fvc, dtype=np.float64)
    pred_sigma = np.array(pred_sigma, dtype=np.float64)

    # Apply clipping to sigma (confidence)
    # The metric requires sigma to be at least 70 ml
    sigma_clipped = np.maximum(pred_sigma, 70)

    # Calculate absolute error (delta)
    raw_delta = np.abs(true_fvc - pred_fvc)

    # Apply thresholding to the error
    # Errors larger than 1000 ml are capped to avoid excessive penalization
    delta = np.minimum(raw_delta, 1000)

    # Calculate the metric
    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the mean metric over the batch
    return np.mean(metric_values)
