import os
import sys
import random
import time
import logging
import numpy as np
from contextlib import contextmanager
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    # Python's built-in random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # OS environment for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (if installed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch is not installed, skip
        pass


def setup_logging(level=logging.INFO):
    """
    Configures the logging setup for the pipeline.

    Args:
        level: The logging level (default: logging.INFO).

    Returns:
        logging.Logger: The configured root logger.
    """
    logger = logging.getLogger()

    # Clear existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


@contextmanager
def timer(name: str):
    """
    Context manager to measure and log the execution time of a block of code.

    Args:
        name (str): The name of the operation being measured.
    """
    t0 = time.time()
    logger = logging.getLogger()
    logger.info(f"[{name}] Start")
    yield
    elapsed = time.time() - t0
    logger.info(f"[{name}] Done in {elapsed:.2f} seconds")


def print_metrics(metrics: dict, prefix: str = ""):
    """
    Prints validation metrics with full precision.

    Args:
        metrics (dict): Dictionary containing metric names and values.
        prefix (str): Optional prefix for the log message.
    """
    logger = logging.getLogger()
    prefix_str = f"{prefix} " if prefix else ""

    log_msg = f"{prefix_str}Metrics: "
    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v}")

    logger.info(log_msg + ", ".join(metric_strs))
