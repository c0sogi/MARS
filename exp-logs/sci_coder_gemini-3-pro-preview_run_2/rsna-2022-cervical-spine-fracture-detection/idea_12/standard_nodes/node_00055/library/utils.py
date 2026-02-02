import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
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


def get_logger(name: str = Config.EXP_NAME, log_file: str = None):
    """
    Creates and returns a logger that outputs to console and optionally a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def calculate_weighted_loss(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the competition weighted multi-label logarithmic loss.

    The metric is defined as:
    L_ij = -w_j * [y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij)]
    Loss is averaged across all rows.

    Weights Assumption:
    - patient_overall: 7.0 (Weighted more highly, balancing the 7 vertebrae)
    - C1 to C7: 1.0 each

    Args:
        y_true (np.ndarray): Binary targets of shape (N, 8).
                             Expected Column Order: [patient_overall, C1, C2, C3, C4, C5, C6, C7]
        y_pred (np.ndarray): Predicted probabilities of shape (N, 8).
        epsilon (float): Clipping value for numerical stability of log.

    Returns:
        float: The calculated weighted log loss.
    """
    # Define weights corresponding to columns: [patient_overall, C1, C2, C3, C4, C5, C6, C7]
    weights = np.array([7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true).astype(np.float64)
    y_pred = np.asarray(y_pred).astype(np.float64)

    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate Binary Cross Entropy terms
    # Term 1: y * log(p)
    term1 = y_true * np.log(y_pred)

    # Term 2: (1-y) * log(1-p)
    term2 = (1 - y_true) * np.log(1 - y_pred)

    # Calculate Weighted Loss per element
    # Weights are broadcasted across the batch dimension
    # Formula: -w_j * [...]
    loss_matrix = -weights * (term1 + term2)

    # Average across all rows (N samples * 8 targets)
    # The competition metric specifies "loss is averaged across all rows".
    total_loss = np.sum(loss_matrix)
    num_rows = y_true.size  # Total number of predictions (N * 8)

    return total_loss / num_rows
