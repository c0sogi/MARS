import os
import sys
import random
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name="logger", log_file=None, level=logging.INFO):
    """
    Configures a logger to output to console and optionally a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger is already configured
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def calculate_accuracy(y_true, y_pred):
    """
    Computes the accuracy classification score.

    Args:
        y_true: 1d array-like, or label indicator array / sparse matrix.
        y_pred: 1d array-like, or label indicator array / sparse matrix.

    Returns:
        float: Accuracy score.
    """
    return accuracy_score(y_true, y_pred)


def calculate_log_loss(y_true, y_pred_proba, labels=None):
    """
    Computes the log loss (cross-entropy loss).

    Args:
        y_true: array-like or label indicator matrix.
        y_pred_proba: array-like of float, predicted probabilities.
        labels: array-like, optional. If not provided, inferred from y_true.

    Returns:
        float: Log loss.
    """
    return log_loss(y_true, y_pred_proba, labels=labels)


def print_metrics(metrics):
    """
    Prints metric values with full precision (no rounding).

    Args:
        metrics: Dictionary of {metric_name: metric_value}.
    """
    for name, value in metrics.items():
        print(f"{name}: {value}")


def save_to_cache(df, path):
    """
    Saves a pandas DataFrame to a parquet file.
    Ensures the directory exists.

    Args:
        df: pandas DataFrame to save.
        path: Destination file path.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_parquet(path, index=False)


def load_from_cache(path):
    """
    Loads a pandas DataFrame from a parquet file.

    Args:
        path: Source file path.

    Returns:
        pandas DataFrame if file exists, else None.
    """
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None
