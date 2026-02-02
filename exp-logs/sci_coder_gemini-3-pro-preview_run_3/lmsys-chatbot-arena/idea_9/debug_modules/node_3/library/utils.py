import os
import sys
import logging
import numpy as np
from sklearn.metrics import log_loss
from library.config import Config, seed_everything


def get_logger(name="main", log_file=None):
    """
    Initializes and configures a logger with both stream and file handlers.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, defaults to Config.output_dir/name.log.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if the logger is already configured
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Stream Handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    if log_file is None:
        os.makedirs(Config.output_dir, exist_ok=True)
        log_file = os.path.join(Config.output_dir, f"{name}.log")

    # Ensure directory for log file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def compute_log_loss(y_true, y_pred):
    """
    Computes the Log Loss metric for the competition.

    Args:
        y_true (array-like): Ground truth labels (probabilities or one-hot).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate log loss with eps='auto' as per metric requirements
    # Sklearn's log_loss handles multiclass and probability targets automatically
    loss = log_loss(y_true, y_pred, eps="auto")

    return loss
