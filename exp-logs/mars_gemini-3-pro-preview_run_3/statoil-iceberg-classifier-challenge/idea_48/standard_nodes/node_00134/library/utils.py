import sys
import logging
import random
from library.config import set_seed as _lib_set_seed, SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Wraps the library implementation to ensure Python's built-in random module
    is also seeded, satisfying the strict reproducibility requirements.

    Args:
        seed (int): The seed value to use. Defaults to SEED from library.config.
    """
    random.seed(seed)
    _lib_set_seed(seed)


def get_logger(name=__name__, level=logging.INFO):
    """
    Configures and returns a logger instance for tracking progress.
    Ensures that handlers are not duplicated if the logger is requested multiple times.

    Args:
        name (str): Name of the logger. Defaults to the module name.
        level (int): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger already has handlers to prevent duplicate logging
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent propagation to the root logger to avoid double printing
        # if the root logger is also configured.
        logger.propagate = False

    return logger
