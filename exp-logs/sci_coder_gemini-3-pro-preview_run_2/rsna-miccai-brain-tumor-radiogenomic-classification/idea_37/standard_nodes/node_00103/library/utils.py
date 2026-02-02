import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN to guarantee reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "MGMT_Pipeline") -> logging.Logger:
    """
    Configures and returns a logger instance for tracking pipeline progress.

    Args:
        name (str): The name of the logger. Defaults to "MGMT_Pipeline".

    Returns:
        logging.Logger: A configured logger instance outputting to stdout.
    """
    logger = logging.getLogger(name)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler that writes to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Define format: Time - Level - Message
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        # Prevent propagation to root logger to avoid double printing if root is configured
        logger.propagate = False

    return logger
