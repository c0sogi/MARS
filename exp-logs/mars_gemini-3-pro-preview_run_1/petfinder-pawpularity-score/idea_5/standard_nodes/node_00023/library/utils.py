import os
import sys
import logging
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to Config.set_seed to ensure consistency with the global configuration.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def compute_rmse(y_true, y_pred):
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.
    Handles both NumPy arrays and PyTorch tensors.

    Args:
        y_true (array-like or torch.Tensor): Ground truth target values.
        y_pred (array-like or torch.Tensor): Estimated target values.

    Returns:
        float: The RMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate RMSE
    # squared parameter was removed in scikit-learn 1.6+
    # We use np.sqrt(mse) for compatibility
    return np.sqrt(mean_squared_error(y_true, y_pred))


def setup_logger(name="Pawpularity", log_file=None, level=logging.INFO):
    """
    Sets up a logger to print messages to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, no file handler is added.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        if log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
