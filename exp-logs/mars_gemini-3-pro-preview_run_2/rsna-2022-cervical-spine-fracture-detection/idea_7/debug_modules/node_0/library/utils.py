import os
import sys
import random
import logging
import numpy as np
import torch
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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name=__name__):
    """
    Creates and configures a logger that outputs to stdout.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger


def competition_metric(y_pred, y_true):
    """
    Calculates the weighted multi-label logarithmic loss used in the competition.

    Formula:
    L_ij = -w_j * [y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij)]
    Loss is averaged across all rows and columns.

    Args:
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, 8).
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, 8).

    Returns:
        float: The calculated weighted log loss.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Ensure inputs are float
    y_pred = y_pred.astype(np.float64)
    y_true = y_true.astype(np.float64)

    # Clip predictions to avoid log(0) and log(1)
    # Using a small epsilon value
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Retrieve weights from Config
    # Config.LOSS_WEIGHTS is a torch.Tensor, convert to numpy
    weights = Config.LOSS_WEIGHTS
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    # Ensure weights shape matches the number of classes (columns)
    # weights shape should be (8,)
    if weights.shape[0] != y_pred.shape[1]:
        raise ValueError(
            f"Shape mismatch: weights {weights.shape} vs preds {y_pred.shape}"
        )

    # Calculate Binary Cross Entropy per element
    # bce = y * log(p) + (1-y) * log(1-p)
    bce = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

    # Apply weights
    # The formula is L = -w * bce
    weighted_loss = -weights * bce

    # Return the mean over all elements
    return np.mean(weighted_loss)
