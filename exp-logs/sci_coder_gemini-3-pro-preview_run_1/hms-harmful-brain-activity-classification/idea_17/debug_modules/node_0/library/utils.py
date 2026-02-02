import os
import sys
import random
import numpy as np
import torch
import logging
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "main"):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def kl_divergence(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15
) -> float:
    """
    Computes the Kullback-Leibler (KL) Divergence between true and predicted probabilities.

    Metric = sum(y_true * log(y_true / y_pred))

    Args:
        y_true (np.ndarray): Ground truth probabilities, shape (N, C).
        y_pred (np.ndarray): Predicted probabilities, shape (N, C).
        epsilon (float): Small value to prevent division by zero or log of zero.

    Returns:
        float: The average KL divergence over the batch.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate KL Divergence
    # D_KL(P || Q) = sum(P * log(P / Q)) = sum(P * log(P) - P * log(Q))

    # Term 1: sum(y_true * log(y_true))
    # Handle 0 * log(0) = 0 using a mask where y_true > 0
    term1 = np.zeros_like(y_true)
    mask = y_true > 0
    term1[mask] = y_true[mask] * np.log(y_true[mask])

    # Term 2: sum(y_true * log(y_pred))
    term2 = y_true * np.log(y_pred)

    # Sum over classes (axis=1)
    kl_per_sample = np.sum(term1 - term2, axis=1)

    # Mean over samples
    return float(np.mean(kl_per_sample))
