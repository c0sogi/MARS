import logging
import sys
from library.config import seed_everything


def get_logger(name: str = "Main"):
    """
    Creates and configures a logger that outputs to stdout.
    Ensures that handlers are not duplicated if the logger is requested multiple times.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers to the same logger if get_logger is called repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Define a consistent format for logs
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid double logging if root is configured elsewhere
    logger.propagate = False

    return logger
