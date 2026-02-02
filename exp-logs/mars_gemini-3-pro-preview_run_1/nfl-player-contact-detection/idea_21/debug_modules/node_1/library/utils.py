import os
import sys
import time
import random
import logging
import warnings
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from library.config import SEED, WORKING_DIR


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def suppress_warnings():
    """
    Suppresses standard warnings to keep the output clean.
    """
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def compute_mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient (MCC).

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.

    Returns:
        float: The MCC score.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return matthews_corrcoef(y_true, y_pred)


def get_logger(name="pipeline", log_file=None):
    """
    Configures and returns a logger that writes to both console and a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file. If None, defaults to
                                  'execution.log' in the WORKING_DIR.

    Returns:
        logging.Logger: The configured logger instance.
    """
    if log_file is None:
        log_file = os.path.join(WORKING_DIR, "execution.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if the logger is retrieved multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not set up file logging to {log_file}: {e}")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


class Timer:
    """
    Context manager to measure and log the execution time of a code block.
    """

    def __init__(self, description="Process", logger=None):
        """
        Args:
            description (str): Description of the process being timed.
            logger (logging.Logger, optional): Logger to use for output. If None, uses print.
        """
        self.description = description
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        msg = f"Starting {self.description}..."
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        msg = (
            f"{self.description} completed in {elapsed_time} seconds."  # Full precision
        )
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
