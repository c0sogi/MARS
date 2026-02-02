import os
import sys
import logging
import numpy as np
from sklearn.metrics import cohen_kappa_score
from library.configuration import seed_everything, Config


def get_logger(name=__name__, log_file=None):
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, only logs to console.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Stream Handler (Console)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def compute_metrics(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.
    Handles regression outputs by clipping to [1, 6] and rounding.

    Args:
        y_true (np.array or torch.Tensor): True scores (integers 1-6).
        y_pred (np.array or torch.Tensor): Predicted scores (continuous).

    Returns:
        dict: Dictionary containing the QWK score.
    """
    # Convert tensors to numpy if necessary
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure arrays are flattened
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    # Post-process regression predictions: Clip to valid range and round
    # The rubric defines scores as integers between 1 and 6
    y_pred_processed = np.clip(y_pred, 1, 6).round().astype(int)
    y_true_processed = y_true.astype(int)

    # Compute Quadratic Weighted Kappa
    qwk = cohen_kappa_score(y_true_processed, y_pred_processed, weights="quadratic")

    return {"qwk": qwk}
