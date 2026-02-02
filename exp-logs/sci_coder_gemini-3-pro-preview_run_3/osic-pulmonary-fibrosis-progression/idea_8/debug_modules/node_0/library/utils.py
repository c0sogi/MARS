import os
import random
import numpy as np
import torch
from library.config import STATS


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def LaplaceLogLikelihood(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric as defined in the task.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: True FVC values (Tensor or numpy array).
        y_pred: Predicted FVC values (Tensor or numpy array).
        sigma: Predicted confidence (sigma) values (Tensor or numpy array).

    Returns:
        float: The mean metric score (negative value, higher is better).
    """
    # Convert numpy arrays to torch tensors if necessary for compatibility
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma)

    # Ensure inputs are on the same device and float type
    device = y_pred.device
    y_true = y_true.to(device).float()
    y_pred = y_pred.to(device).float()
    sigma = sigma.to(device).float()

    # Constants
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device))

    # Clipping logic per metric definition
    # Confidence clipped at 70 ml
    sigma_clipped = torch.clamp(sigma, min=70)

    # Error thresholded at 1000 ml
    delta = torch.abs(y_true - y_pred)
    delta = torch.clamp(delta, max=1000)

    # Metric calculation
    metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)

    return torch.mean(metric).item()


class InverseScaler:
    """
    Utility class to transform standardized predictions back to the original scale
    using statistics defined in the configuration.
    """

    def __init__(self):
        self.fvc_mean = STATS["FVC_MEAN"]
        self.fvc_std = STATS["FVC_STD"]
        self.weeks_mean = STATS["WEEKS_MEAN"]
        self.weeks_std = STATS["WEEKS_STD"]

    def inverse_scale_fvc(self, fvc_scaled):
        """
        Reverts Z-score normalization for FVC.
        Formula: x = z * std + mean
        """
        return fvc_scaled * self.fvc_std + self.fvc_mean

    def inverse_scale_weeks(self, weeks_scaled):
        """
        Reverts Z-score normalization for Weeks.
        Formula: x = z * std + mean
        """
        return weeks_scaled * self.weeks_std + self.weeks_mean

    def inverse_scale_sigma(self, sigma_scaled):
        """
        Reverts Z-score normalization for Uncertainty (Sigma).
        Since Sigma represents a magnitude/spread, we only scale by the standard deviation.
        Formula: sigma = sigma_scaled * std
        """
        return sigma_scaled * self.fvc_std
