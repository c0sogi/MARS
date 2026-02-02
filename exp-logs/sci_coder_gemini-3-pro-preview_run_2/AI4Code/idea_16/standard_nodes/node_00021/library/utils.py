import os
import sys
import time
import random
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_STATE.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    """
    Context manager to measure and print the execution time of a code block.
    """

    def __init__(self, name: str = "Process"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.name}] Elapsed time: {elapsed_time:.4f}s")


def get_logger(name: str = "main"):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if the logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
