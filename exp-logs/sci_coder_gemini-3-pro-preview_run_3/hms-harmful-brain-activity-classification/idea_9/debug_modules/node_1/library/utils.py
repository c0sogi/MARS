import os
import sys
import random
import numpy as np
import torch
import logging


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Useful for tracking loss and accuracy during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The current value to record.
            n (int): The weight or number of items associated with this value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(filename: str) -> logging.Logger:
    """
    Initializes and returns a logger that outputs to both console and a file.

    Args:
        filename (str): The path to the log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Create logger with a specific name to avoid interference
    logger = logging.getLogger("experiment_logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs if initialized multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # File Handler: Includes timestamp and level
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Console Handler: Plain message for readability
    console_formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger
