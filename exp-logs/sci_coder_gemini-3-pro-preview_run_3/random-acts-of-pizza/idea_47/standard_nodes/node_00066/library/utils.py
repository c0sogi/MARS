import os
import sys
import logging
import time
from contextlib import contextmanager
from library.config import set_seed

# Re-export set_seed for convenience and to satisfy the module description
__all__ = ["set_seed", "setup_logging", "get_logger", "timer"]


def setup_logging(log_path=None, level=logging.INFO):
    """
    Configures the global logging configuration.

    Args:
        log_path (str, optional): Path to the log file. If provided, logs will be written to this file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_path:
        # Ensure the directory for the log file exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    # Configure the root logger
    # force=True ensures we override any existing configuration
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name):
    """
    Retrieves a logger with the specified name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)


@contextmanager
def timer(name):
    """
    Context manager to measure and log the execution time of a code block.

    Args:
        name (str): Name of the operation being measured.
    """
    t0 = time.time()
    logger = get_logger(name)
    logger.info(f"[{name}] Start")
    try:
        yield
    finally:
        elapsed = time.time() - t0
        logger.info(f"[{name}] Done in {elapsed:.2f} s")
