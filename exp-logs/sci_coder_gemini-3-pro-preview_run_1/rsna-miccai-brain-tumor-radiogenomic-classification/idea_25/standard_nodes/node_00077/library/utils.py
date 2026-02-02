import os
import sys
import random
import time
import logging
import numpy as np
import torch
from contextlib import contextmanager
from library import config


def seed_everything(seed: int = config.SEED):
    """
    Seeds all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the PyTorch device to be used for training/inference.

    Returns:
        torch.device: The device specified in config.DEVICE.
    """
    return torch.device(config.DEVICE)


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Creates and configures a logger that outputs to console and optionally a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


@contextmanager
def timer(name: str, logger: logging.Logger = None):
    """
    Context manager to measure and log the execution time of a code block.

    Args:
        name (str): Name of the operation being timed.
        logger (logging.Logger, optional): Logger to use. If None, prints to stdout.
    """
    t0 = time.time()
    msg_start = f"[{name}] Start"
    if logger:
        logger.info(msg_start)
    else:
        print(msg_start)

    yield

    elapsed = time.time() - t0
    msg_end = f"[{name}] Done in {elapsed:.2f} s"
    if logger:
        logger.info(msg_end)
    else:
        print(msg_end)
