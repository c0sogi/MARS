import os
import random
import logging
import numpy as np
import torch
from sklearn.metrics import log_loss, accuracy_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name="main", log_file=None):
    """
    Creates and configures a logger.

    Args:
        name (str): Name of the logger.
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
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def compute_metrics(predictions, targets):
    """
    Computes Log Loss and Accuracy for the competition.

    Args:
        predictions (np.ndarray): Predicted probabilities of shape (N, 3).
        targets (np.ndarray): Ground truth targets. Can be shape (N, 3) for soft labels/probabilities
                              or (N,) for hard class indices.

    Returns:
        dict: A dictionary containing 'log_loss' and 'accuracy'.
    """
    # Calculate Log Loss
    # sklearn.metrics.log_loss handles clipping internally (eps)
    try:
        loss = log_loss(targets, predictions)
    except Exception as e:
        # Fallback or error handling if needed, though inputs should be valid
        print(f"Error computing log loss: {e}")
        loss = float("nan")

    # Calculate Accuracy
    # For accuracy, we need hard labels.
    # If targets are probabilities (soft labels), take argmax.
    if targets.ndim > 1:
        true_indices = np.argmax(targets, axis=1)
    else:
        true_indices = targets

    pred_indices = np.argmax(predictions, axis=1)
    acc = accuracy_score(true_indices, pred_indices)

    return {"log_loss": loss, "accuracy": acc}
