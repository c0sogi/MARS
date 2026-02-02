import sys
import logging
import torch
from library.config import set_seed


def get_device() -> torch.device:
    """
    Selects and returns the appropriate PyTorch device.
    Returns 'cuda' if a GPU is available, otherwise returns 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str = "RSNA_MGMT") -> logging.Logger:
    """
    Configures and returns a logger with a stream handler for stdout.
    Ensures that the logger is only configured once to avoid duplicate logs.
    """
    logger = logging.getLogger(name)

    # Check if the logger already has handlers to prevent duplicate logging
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler that logs to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Define a simple format
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
