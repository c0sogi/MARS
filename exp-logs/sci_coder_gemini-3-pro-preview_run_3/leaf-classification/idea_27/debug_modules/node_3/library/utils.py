import os
import sys
import random
import logging
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_path: str = None, level=logging.INFO):
    """
    Configures the logging system.

    Args:
        log_path (str, optional): Path to the log file. If None, logs only to stdout.
        level (int): Logging level.
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_path:
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    # Reset existing handlers to prevent duplicate logs if setup_logging is called multiple times
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def print_metric(name: str, value: float):
    """
    Prints a metric name and its value with full precision.

    Args:
        name (str): Name of the metric.
        value (float): Value of the metric.
    """
    # Using f-string without formatting specifier ensures full precision
    print(f"{name}: {value}")
