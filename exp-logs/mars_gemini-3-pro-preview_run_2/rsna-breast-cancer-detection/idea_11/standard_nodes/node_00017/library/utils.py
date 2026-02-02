import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth labels (0 or 1).
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities (0 to 1).
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The probabilistic F1 score.
    """
    # Convert to torch tensors if they are numpy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Ensure float types and same device
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Calculate Probabilistic True Positives (pTP)
    # Sum of probabilities for the positive class instances
    p_tp = (y_true * y_pred).sum()

    # Calculate Probabilistic False Positives (pFP)
    # Sum of probabilities for negative class instances incorrectly predicted as positive
    p_fp = ((1 - y_true) * y_pred).sum()

    # Calculate Total Positives (TP + FN)
    # This is simply the count of actual positive labels
    total_positives = y_true.sum()

    # Calculate pPrecision: pTP / (pTP + pFP)
    # Note: pTP + pFP = sum(y_pred)
    p_precision = p_tp / (p_tp + p_fp + epsilon)

    # Calculate pRecall: pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # Calculate pF1
    pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return pf1.item()


def get_logger(name, log_file=None):
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, logs only to console.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_device():
    """
    Returns the appropriate PyTorch device based on Config and availability.
    """
    return torch.device(Config.DEVICE)
