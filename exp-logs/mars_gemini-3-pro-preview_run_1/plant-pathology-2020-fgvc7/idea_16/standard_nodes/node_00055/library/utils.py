import os
import random
import numpy as np
import torch
import logging
import sys
from library.config import Config


def seed_everything(seed: int = 42):
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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Determines the available computational device.

    Returns:
        torch.device: The device (cuda or cpu).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def save_model(model: torch.nn.Module, path: str):
    """
    Saves the model's state dictionary to the specified path.
    Ensures the parent directory exists.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path where the model state dict will be saved.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save only the state dict for efficiency and flexibility
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: torch.device = None):
    """
    Loads a state dictionary into the provided model architecture.

    Args:
        model (torch.nn.Module): The model architecture instance.
        path (str): The file path to the saved state dict.
        device (torch.device, optional): The device to map the location to.
                                         Defaults to the result of get_device().

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    if device is None:
        device = get_device()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def get_logger(
    name: str = "apple_disease_detection", log_file: str = None
) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to a file to log to. If None, logs only to stdout.
                                  Defaults to a file in the Config.WORKING_DIR.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file is None:
        # Default log file in working directory
        log_file = os.path.join(Config.WORKING_DIR, "training.log")

    # Ensure directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
