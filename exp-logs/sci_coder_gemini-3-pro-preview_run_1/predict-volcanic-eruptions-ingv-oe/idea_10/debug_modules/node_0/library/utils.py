import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
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
    torch.cuda.manual_seed_all(seed)

    # deterministic=True ensures reproducible results but may impact performance
    torch.backends.cudnn.deterministic = True
    # benchmark=False prevents the auto-tuner from selecting different algorithms
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "main", log_file: str = None):
    """
    Configures and returns a logger instance for consistent logging.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If provided, logs will be written to this file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging if get_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Output to Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler (Output to File)
    if log_file:
        try:
            # Ensure the directory for the log file exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to create log file handler: {e}")

    return logger
