import os
import random
import numpy as np
import torch
import logging
import shutil
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations ensure that running the same code on the same hardware
    # produces the same results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Creates and configures a logger that writes to both console and a file.

    Args:
        name (str): The name of the logger.
        log_file (str): Path to the log file. If None, defaults to a file in WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file is None:
        log_file = os.path.join(Config.WORKING_DIR, "training.log")

    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: str = Config.CHECKPOINT_DIR,
    filename: str = "checkpoint.pth",
):
    """
    Saves the model state dictionary to a file.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: str = Config.DEVICE,
):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (optional): The scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch or best_score).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the Log Loss metric.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The calculated log loss.
    """
    # sklearn's log_loss handles clipping internally (eps=1e-15 by default)
    # ensuring numerical stability.
    loss = log_loss(y_true, y_pred)
    return loss
