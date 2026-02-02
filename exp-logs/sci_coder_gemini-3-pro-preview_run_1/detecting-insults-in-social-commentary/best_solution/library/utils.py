import os
import random
import numpy as np
import torch
import logging
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename: str = os.path.join(Config.working_dir, "train.log")):
    """
    Initializes and returns a logger that outputs to both a file and the console.

    Args:
        filename (str): Path to the log file. Defaults to 'train.log' in the working directory.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(filename)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers to the same logger if function is called repeatedly
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


def get_auc_score(y_true, y_pred):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores or probabilities.

    Returns:
        float: The ROC AUC score.
    """
    return roc_auc_score(y_true, y_pred)
