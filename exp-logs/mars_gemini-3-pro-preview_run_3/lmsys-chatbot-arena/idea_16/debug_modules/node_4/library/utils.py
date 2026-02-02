import os
import random
import logging
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
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


def get_logger(name="Main"):
    """
    Configures and returns a logger instance for tracking experiment progress.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Only add handler if one doesn't exist to prevent duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def compute_log_loss(y_true, y_pred):
    """
    Computes the Log Loss metric using sklearn.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels (N, 3).
                                             Can be probabilities or one-hot encoded.
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities (N, 3).

    Returns:
        float: The calculated log loss.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float64 for precision
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # Calculate log loss
    # eps is handled automatically by sklearn (default 1e-15)
    score = log_loss(y_true, y_pred)

    return score
