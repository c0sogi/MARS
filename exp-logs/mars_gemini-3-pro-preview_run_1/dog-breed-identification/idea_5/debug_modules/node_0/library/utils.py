import os
import sys
import random
import numpy as np
import torch
import logging
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Configures CuDNN for deterministic execution.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior in CuDNN
    # This might impact performance slightly but is required for exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "dog_breed_clf", log_file: str = None):
    """
    Creates and configures a logger for structured logging.
    Writes logs to both the console and a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If None, uses 'train.log' in Config.WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is None:
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        log_file = os.path.join(Config.WORKING_DIR, "train.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the logger is retrieved multiple times
    if not logger.handlers:
        # Define format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File Handler
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file at {log_file}. Error: {e}")

        # Stream Handler (Stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics like loss during training epochs.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
