import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name="project_logger", log_file=None):
    """
    Creates and returns a logger that logs to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file. If None, defaults to 'train.log' in Config.WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(Config.WORKING_DIR, "train.log")

    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Computes Log Loss and Accuracy for binary classification.

    Args:
        y_true (np.array or list): Ground truth binary labels (0 or 1).
        y_pred_prob (np.array or list): Predicted probabilities for class 1.
        threshold (float): Threshold for converting probabilities to binary predictions for accuracy calculation.

    Returns:
        dict: A dictionary containing 'log_loss' and 'accuracy'.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    # Calculate Log Loss
    # Scikit-learn's log_loss handles clipping internally (default eps=1e-15)
    loss = log_loss(y_true, y_pred_prob)

    # Calculate Accuracy
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)

    return {"log_loss": loss, "accuracy": acc}
