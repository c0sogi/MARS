import os
import sys
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility by delegating to Config.set_seed.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    Config.set_seed(seed)


def get_score(y_true, y_pred):
    """
    Calculates the average Area Under the ROC Curve (AUC) for each label.

    This function iterates through each class column and calculates the ROC AUC.
    It handles cases where a class might only have one unique value in the
    provided batch (which would otherwise raise a ValueError in sklearn) by
    skipping that column for the average.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The average AUC score across all valid columns. Returns 0.0 if no columns are valid.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    aucs = []
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        # Only calculate AUC if the column has both classes (0 and 1) present
        # roc_auc_score requires both positive and negative samples
        if len(np.unique(y_true[:, i])) > 1:
            try:
                score = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(score)
            except ValueError:
                # In case of any other sklearn validation error, skip this column
                pass

    if len(aucs) == 0:
        return 0.0

    return np.mean(aucs)


def get_logger(name="train"):
    """
    Configures and returns a logger instance that writes to stdout and a log file.

    Args:
        name (str): The name of the logger. Defaults to "train".

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to prevent duplicate logs
    if not logger.handlers:
        # Stream Handler (Output to console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # File Handler (Output to file in working directory)
        # Ensure the working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        log_file_path = os.path.join(Config.WORKING_DIR, f"{name}.log")

        file_handler = logging.FileHandler(log_file_path)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
