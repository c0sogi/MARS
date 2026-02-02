import os
import random
import sys
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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = Config.IDEA_NAME):
    """
    Creates and configures a logger that outputs to stdout.

    Args:
        name (str): The name of the logger. Defaults to Config.IDEA_NAME.

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        # Simple format: Time - Level - Message
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all internal statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter with a new value.

        Args:
            val (float): The value to add.
            n (int): The weight/count of the value (default 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clean_text(text: str) -> str:
    """
    Basic text cleaning helper.
    Removes leading/trailing whitespace and normalizes internal spacing.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text)
    return " ".join(text.strip().split())
