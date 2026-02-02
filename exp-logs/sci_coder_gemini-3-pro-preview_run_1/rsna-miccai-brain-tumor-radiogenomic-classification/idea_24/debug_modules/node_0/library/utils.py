import os
import sys
import logging
import torch
from library.config import Config, set_seed


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the implementation provided in library.config.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    set_seed(seed)


def get_device() -> torch.device:
    """
    Retrieves the PyTorch device configuration.

    Returns:
        torch.device: The device object (CPU or CUDA) as defined in Config.
    """
    return torch.device(Config.DEVICE)


def setup_logger(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configures the global logger with console and optional file output.

    Args:
        log_file (str, optional): Path to the file where logs should be saved.
                                  If None, logs are only printed to console.
        level (int): The logging threshold level (e.g., logging.INFO).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger("MGMT_Classification")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplication if setup is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (Stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Optional)
    if log_file:
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
