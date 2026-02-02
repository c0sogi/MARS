import os
import sys
import random
import numpy as np
import torch
import logging
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Macro-Averaged Area Under the ROC Curve (AUC).

    Args:
        y_true: Ground truth labels (binary), shape (n_samples, n_classes).
                Can be numpy array or torch Tensor.
        y_pred: Predicted probabilities, shape (n_samples, n_classes).
                Can be numpy array or torch Tensor.

    Returns:
        float: The ROC AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    try:
        # Standard macro-averaged ROC AUC
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # This block handles edge cases where a specific class might have only
        # one unique label (e.g., all 0s) in the provided batch/split.
        # Scikit-learn's roc_auc_score throws a ValueError in this case.
        aucs = []
        for i in range(y_true.shape[1]):
            # Calculate AUC only for columns with both positive and negative samples
            if len(np.unique(y_true[:, i])) > 1:
                aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

        if len(aucs) == 0:
            score = 0.5
        else:
            score = np.mean(aucs)

    if np.isnan(score):
        return 0.5
    return score


def get_logger(log_file: str):
    """
    Creates and configures a logger that writes to both a file and the console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    # File Handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
