import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=None):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int, optional): The seed value to use. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_global_stats(csv_path=None):
    """
    Calculates the mean and standard deviation of the FVC target from the training data.

    Args:
        csv_path (str, optional): Path to the training CSV. Defaults to Config.TRAIN_CSV.

    Returns:
        tuple: (mean, std) of the FVC column.
    """
    if csv_path is None:
        csv_path = Config.TRAIN_CSV

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training metadata not found at {csv_path}")

    df = pd.read_csv(csv_path)
    mean_fvc = df["FVC"].mean()
    std_fvc = df["FVC"].std()

    return mean_fvc, std_fvc


def inverse_transform(mu_scaled, sigma_scaled, target_mean=None, target_std=None):
    """
    Converts standardized predictions back to the original scale (ml) and applies
    competition-specific post-processing (clipping sigma).

    Args:
        mu_scaled (np.ndarray or torch.Tensor): Predicted standardized mean.
        sigma_scaled (np.ndarray or torch.Tensor): Predicted standardized standard deviation.
        target_mean (float, optional): Global mean used for normalization. Defaults to Config.TARGET_MEAN.
        target_std (float, optional): Global std used for normalization. Defaults to Config.TARGET_STD.

    Returns:
        tuple: (mu_original, sigma_original) in ml.
    """
    if target_mean is None:
        target_mean = Config.TARGET_MEAN
    if target_std is None:
        target_std = Config.TARGET_STD

    # Handle both Tensor and NumPy inputs
    is_tensor = torch.is_tensor(mu_scaled)

    if is_tensor:
        mu = mu_scaled * target_std + target_mean
        sigma = sigma_scaled * target_std
        # Apply hard clip for submission requirements: max(sigma, 70)
        sigma = torch.clamp(sigma, min=70.0)
    else:
        mu = mu_scaled * target_std + target_mean
        sigma = sigma_scaled * target_std
        # Apply hard clip for submission requirements: max(sigma, 70)
        sigma = np.maximum(sigma, 70.0)

    return mu, sigma


def calculate_metric(fvc_true, fvc_pred, sigma_pred):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        score = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        fvc_true (np.ndarray): Ground truth FVC values.
        fvc_pred (np.ndarray): Predicted FVC values.
        sigma_pred (np.ndarray): Predicted confidence (sigma).

    Returns:
        float: The average metric score.
    """
    # Ensure inputs are numpy arrays
    if torch.is_tensor(fvc_true):
        fvc_true = fvc_true.detach().cpu().numpy()
    if torch.is_tensor(fvc_pred):
        fvc_pred = fvc_pred.detach().cpu().numpy()
    if torch.is_tensor(sigma_pred):
        sigma_pred = sigma_pred.detach().cpu().numpy()

    sigma_clipped = np.maximum(sigma_pred, 70)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000)

    metric = -(np.sqrt(2) * delta) / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)

    return np.mean(metric)
