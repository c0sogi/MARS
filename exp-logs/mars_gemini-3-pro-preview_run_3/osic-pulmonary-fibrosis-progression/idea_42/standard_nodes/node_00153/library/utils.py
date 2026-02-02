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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood_metric(true_fvc, pred_fvc, pred_sigma):
    """
    Calculates the modified Laplace Log Likelihood metric used in the competition.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        true_fvc (np.array or torch.Tensor): Ground truth FVC values.
        pred_fvc (np.array or torch.Tensor): Predicted FVC values.
        pred_sigma (np.array or torch.Tensor): Predicted confidence (sigma).

    Returns:
        float: The mean metric score over the batch.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(true_fvc, torch.Tensor):
        true_fvc = true_fvc.detach().cpu().numpy()
    if isinstance(pred_fvc, torch.Tensor):
        pred_fvc = pred_fvc.detach().cpu().numpy()
    if isinstance(pred_sigma, torch.Tensor):
        pred_sigma = pred_sigma.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    true_fvc = np.array(true_fvc, dtype=np.float64)
    pred_fvc = np.array(pred_fvc, dtype=np.float64)
    pred_sigma = np.array(pred_sigma, dtype=np.float64)

    # Clipping sigma
    sigma_clipped = np.maximum(pred_sigma, Config.SIGMA_CLIP)

    # Calculating delta with clipping
    delta = np.abs(true_fvc - pred_fvc)
    delta = np.minimum(delta, Config.MAX_ERROR)

    # Metric calculation
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)


def inverse_transform_predictions(pred_mu_scaled, pred_sigma_scaled):
    """
    Inverse transforms the model predictions from Z-score normalized space
    back to the original FVC scale (ml).

    Args:
        pred_mu_scaled (np.array or torch.Tensor): Normalized predicted mean.
        pred_sigma_scaled (np.array or torch.Tensor): Normalized predicted std dev.

    Returns:
        tuple: (pred_mu_original, pred_sigma_original) in ml.
    """
    # Convert tensors to numpy arrays if necessary for consistency,
    # though simple arithmetic works on tensors too.
    # We keep them as is if they are tensors to allow gradient flow if needed,
    # or simple calculation if numpy.

    # Inverse Z-score scaling: x = z * std + mean
    pred_mu_original = pred_mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN

    # Standard deviation scales multiplicatively: sigma = z_sigma * std
    pred_sigma_original = pred_sigma_scaled * Config.TARGET_STD

    return pred_mu_original, pred_sigma_original
