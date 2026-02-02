import os
import random
import logging
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
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


def get_logger(name, filename=None):
    """
    Creates and returns a logger instance.

    Args:
        name (str): The name of the logger.
        filename (str, optional): Path to a file to log to.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if function is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if filename:
            file_handler = logging.FileHandler(filename)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_device():
    """
    Returns the PyTorch device configured in Config.

    Returns:
        torch.device: The device (cpu or cuda).
    """
    return torch.device(Config.DEVICE)


def calculate_weighted_log_loss(y_true, y_pred):
    """
    Calculates the weighted multi-label logarithmic loss according to the competition structure.

    The metric weights the 'patient_overall' label equally to the sum of the 7 vertebral labels.
    Weights:
        - patient_overall: 1.0
        - C1-C7: 1/7 each

    Args:
        y_true (pd.DataFrame): DataFrame containing true binary labels.
                               Must contain columns: ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'patient_overall'].
        y_pred (pd.DataFrame): DataFrame containing predicted probabilities.
                               Must contain columns: ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'patient_overall'].

    Returns:
        float: The calculated weighted log loss.
    """
    # Define weights based on the task description and solution strategy
    # Sum of weights = 1.0 + 7 * (1/7) = 2.0
    weights = {
        "patient_overall": 1.0,
        "C1": 1.0 / 7.0,
        "C2": 1.0 / 7.0,
        "C3": 1.0 / 7.0,
        "C4": 1.0 / 7.0,
        "C5": 1.0 / 7.0,
        "C6": 1.0 / 7.0,
        "C7": 1.0 / 7.0,
    }

    # Validate input columns
    required_cols = list(weights.keys())
    for col in required_cols:
        if col not in y_true.columns:
            raise ValueError(f"Missing column '{col}' in y_true.")
        if col not in y_pred.columns:
            raise ValueError(f"Missing column '{col}' in y_pred.")

    total_loss = 0.0
    total_weight = 0.0
    epsilon = 1e-15

    for col, w in weights.items():
        # Extract data
        yt = y_true[col].values.astype(float)
        yp = y_pred[col].values.astype(float)

        # Clip probabilities to avoid log(0)
        yp = np.clip(yp, epsilon, 1 - epsilon)

        # Calculate Binary Cross Entropy for this label
        # Log loss = -[y * log(p) + (1-y) * log(1-p)]
        bce = -(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))

        # Average over the dataset
        mean_bce = np.mean(bce)

        # Accumulate weighted loss
        total_loss += w * mean_bce
        total_weight += w

    # Normalize by the sum of weights
    weighted_log_loss = total_loss / total_weight

    return weighted_log_loss
