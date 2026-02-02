import os
import sys
import torch
import logging
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the centralized Config.set_seed method to ensure consistency.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def save_checkpoint(model: torch.nn.Module, path: str):
    """
    Saves the model's state dictionary to the specified file path.
    Automatically creates the parent directory if it does not exist.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): The destination file path for the checkpoint.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), path)


def load_checkpoint(
    model: torch.nn.Module, path: str, device: str = Config.DEVICE
) -> torch.nn.Module:
    """
    Loads the model's state dictionary from the specified file path.

    Args:
        model (torch.nn.Module): The PyTorch model instance to load weights into.
        path (str): The file path of the checkpoint.
        device (str): The device to map the location to (default: Config.DEVICE).

    Returns:
        torch.nn.Module: The model with loaded weights.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def get_logger(name: str = "CG-WBN") -> logging.Logger:
    """
    Configures and returns a logger for the application.
    Ensures logs are printed to stdout without extra formatting or progress bars.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Use a simple message format: just the message, no timestamps/levels
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_metrics(
    logger: logging.Logger,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_acc: float = None,
):
    """
    Logs training and validation metrics with full precision.

    Args:
        logger (logging.Logger): The logger instance.
        epoch (int): Current epoch number.
        train_loss (float): Training loss value.
        val_loss (float): Validation loss value.
        val_acc (float, optional): Validation accuracy value.
    """
    msg = f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}"
    if val_acc is not None:
        msg += f", Val Acc = {val_acc}"
    logger.info(msg)
