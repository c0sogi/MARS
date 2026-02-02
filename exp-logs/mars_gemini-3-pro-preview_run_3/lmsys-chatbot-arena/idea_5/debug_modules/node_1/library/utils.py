import os
import sys
import random
import numpy as np
import torch
import logging
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
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

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name="train"):
    """
    Configures and returns a logger that writes to both console and a file.

    Args:
        name (str): The name of the logger. Defaults to "train".

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers if they already exist
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        log_path = os.path.join(Config.OUTPUT_DIR, f"{name}.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def compute_metrics(y_true, y_pred):
    """
    Computes the Log Loss metric for the competition.

    Args:
        y_true (np.array or list): Ground truth probabilities or one-hot labels. Shape (N, 3).
        y_pred (np.array or list): Predicted probabilities. Shape (N, 3).

    Returns:
        dict: A dictionary containing the 'log_loss' score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate Log Loss
    # sklearn.metrics.log_loss handles multiclass probability targets natively
    score = log_loss(y_true, y_pred)

    return {"log_loss": score}
