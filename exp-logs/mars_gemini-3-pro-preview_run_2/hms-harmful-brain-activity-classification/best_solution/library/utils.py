import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across random, numpy, and torch (CPU and GPU).

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


def kl_divergence_score(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the Kullback-Leibler Divergence between true and predicted probabilities.
    Metric = sum(y_true * log(y_true / y_pred))

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth probabilities. Shape (N, C).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities. Shape (N, C).
        epsilon (float): Small value to clip predictions and avoid log(0).

    Returns:
        float: The average KL divergence score across samples.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate KL Divergence: sum(P * log(P / Q))
    # = sum(P * log(P)) - sum(P * log(Q))

    # Handle log(0) in y_true safely
    # If y_true is 0, y_true * log(y_true) is 0 by limit
    with np.errstate(divide="ignore", invalid="ignore"):
        log_true = np.log(y_true)
        # Term 1: P * log(P) -> 0 where P=0
        term1 = np.where(y_true > 0, y_true * log_true, 0.0)

        log_pred = np.log(y_pred)
        # Term 2: P * log(Q)
        term2 = y_true * log_pred

        # Sum over classes (axis=1)
        row_kl = np.sum(term1 - term2, axis=1)

    # Mean over batch (axis=0)
    return np.mean(row_kl)


def get_full_path(relative_path: str) -> str:
    """
    Resolves the full file path by combining the input directory from Config
    with the relative path provided in the metadata.

    Args:
        relative_path (str): The relative path (e.g., 'train_eegs/123.parquet').

    Returns:
        str: The absolute or full relative path to the file.
    """
    if pd.isna(relative_path):
        return None
    return os.path.join(Config.INPUT_DIR, relative_path)
