import os
import sys
import random
import numpy as np
import torch
import logging
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Ensures deterministic behavior for CUDA operations.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name="main"):
    """
    Creates and configures a logger that writes to both console and file.
    The log file is saved in the Config.output_dir.
    """
    # Ensure the output directory exists
    os.makedirs(Config.output_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if the logger is retrieved multiple times
    if not logger.handlers:
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        log_file_path = os.path.join(Config.output_dir, f"{name}.log")
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def compute_metrics(y_true, y_pred):
    """
    Computes Log Loss for the predictions.

    Args:
        y_true: Ground truth labels (soft probabilities or one-hot).
                Shape (N, 3) or (N,).
        y_pred: Predicted probabilities. Shape (N, 3).

    Returns:
        dict: Dictionary containing the 'log_loss'.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate Log Loss
    # sklearn.metrics.log_loss handles eps automatically (default 1e-15)
    # It also handles soft targets (probability distributions)
    loss = log_loss(y_true, y_pred)

    return {"log_loss": loss}


def get_device():
    """
    Returns the appropriate torch device (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
