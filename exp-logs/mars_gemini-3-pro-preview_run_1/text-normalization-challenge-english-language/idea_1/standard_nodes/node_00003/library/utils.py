import sys
import logging
import time
from contextlib import contextmanager
from library.config import Config, set_seed


def get_logger(name: str = "app") -> logging.Logger:
    """
    Configures and returns a logger that prints to stdout.
    Ensures no duplicate handlers are added to prevent repeated log messages.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if the logger is retrieved multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid double logging if root is configured
    logger.propagate = False

    return logger


@contextmanager
def timer(task_name: str, logger: logging.Logger = None):
    """
    Context manager to measure and log execution time of a code block.

    Args:
        task_name (str): Description of the task being timed.
        logger (logging.Logger, optional): Logger to use for output.
                                           If None, prints to stdout.
    """
    start_time = time.time()
    msg_start = f"Starting {task_name}..."

    if logger:
        logger.info(msg_start)
    else:
        print(msg_start)

    yield

    elapsed_time = time.time() - start_time
    msg_end = f"{task_name} completed in {elapsed_time:.6f} seconds"

    if logger:
        logger.info(msg_end)
    else:
        print(msg_end)
