import sys
import logging
from library import config


def seed_everything(seed: int = config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the centralized configuration implementation.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    config.set_seed(seed)


def get_logger(name: str = "EXP"):
    """
    Creates and returns a standardized logger configured to write to stdout.
    Ensures handlers are not duplicated if the logger is requested multiple times.

    Args:
        name (str): The name of the logger. Defaults to "EXP".

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to prevent duplicate logging
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Define a standard format including timestamp, logger name, level, and message
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        # Prevent propagation to root logger to avoid double printing if root is configured
        logger.propagate = False

    return logger
