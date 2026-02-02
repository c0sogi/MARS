import os
import random
import numpy as np
import torch
import pandas as pd
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def kl_divergence(y_true, y_pred, epsilon: float = 1e-15):
    """
    Calculates the Kullback-Leibler Divergence between true and predicted probabilities.

    This function expects inputs to be the probability distributions for the classes.
    It handles numerical stability by clipping predictions and treating 0*log(0) as 0.

    Args:
        y_true (pd.DataFrame or np.ndarray): Ground truth probabilities (N, C).
        y_pred (pd.DataFrame or np.ndarray): Predicted probabilities (N, C).
        epsilon (float): Small value to clip predictions and avoid log(0).

    Returns:
        float: The average KL divergence score across all samples.
    """
    # Convert pandas objects to numpy arrays
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # Ensure inputs are float
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # Clip predictions to prevent division by zero or log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate KL Divergence: sum(P(x) * log(P(x) / Q(x)))
    # We use np.where or masking to handle the case where y_true is 0.
    # Mathematically, lim(x->0) x*log(x) = 0.

    # Calculate the log term: log(y_true / y_pred)
    # Since y_pred is clipped, division is safe.
    # If y_true is 0, log(0) is -inf, but we will multiply by 0 later.
    # To avoid RuntimeWarning for log(0), we can use a safe mask.

    # Calculate terms
    # We rely on numpy's handling: 0 * -inf = nan, then we replace nan with 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = y_true * np.log(y_true / y_pred)

    # Replace NaNs (resulting from 0 * log(0)) with 0.0
    terms = np.nan_to_num(terms, nan=0.0)

    # Sum over classes (axis=1) to get KL for each sample
    sample_kl = np.sum(terms, axis=1)

    # Return the mean over all samples
    return np.mean(sample_kl)
