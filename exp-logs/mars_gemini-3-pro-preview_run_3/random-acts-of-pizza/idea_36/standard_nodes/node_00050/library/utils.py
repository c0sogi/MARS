import os
import sys
import random
import logging
import time
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    # Python random
    random.seed(seed)

    # Environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Deterministic algorithms for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Note: Pandas relies on NumPy's random generator, so np.random.seed covers it.


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger with a standard format.

    Args:
        name (str): The name of the logger (usually __name__).
        level (int): The logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def print_metric(name: str, value: float):
    """
    Prints a metric with full precision as required by the task.

    Args:
        name (str): The name of the metric (e.g., "Validation AUC").
        value (float): The value of the metric.
    """
    print(f"{name}: {value}")


class Timer:
    """
    Context manager to measure and log execution time of code blocks.
    """

    def __init__(self, name: str, logger: logging.Logger = None):
        self.name = name
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        msg = f"Finished {self.name}. Duration: {elapsed_time:.4f} seconds."
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
