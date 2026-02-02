import os
import random
import numpy as np
import torch
import logging
from sklearn.metrics import log_loss
from library.config import DEVICE


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(name, log_file, level=logging.INFO):
    """
    Configures and returns a logger instance that writes to both file and console.

    Args:
        name (str): The name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    # Ensure directory for log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times if the logger is retrieved again
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the binary cross-entropy log loss.

    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_pred (array-like): Predicted probabilities (between 0 and 1).

    Returns:
        float: The log loss value.
    """
    # sklearn's log_loss handles clipping internally (eps=1e-15 by default)
    # to avoid log(0).
    return log_loss(y_true, y_pred)


def save_checkpoint(state, filename):
    """
    Saves the model state to a checkpoint file.

    Args:
        state (dict): The state dictionary containing model parameters, optimizer state, etc.
        filename (str): Path where the checkpoint will be saved.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, device=DEVICE):
    """
    Loads a model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (torch.device): The device to map the checkpoint to.

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    checkpoint = torch.load(filename, map_location=device)

    # Load model state
    # We assume strict=True to ensure architecture matches
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided and present in checkpoint
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
