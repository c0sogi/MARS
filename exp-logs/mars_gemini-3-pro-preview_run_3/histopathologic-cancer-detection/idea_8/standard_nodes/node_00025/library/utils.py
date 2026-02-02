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
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_auc(y_true, y_pred):
    """
    Computes the Area Under the ROC Curve (AUC).

    Args:
        y_true: Ground truth labels (binary). Can be list, numpy array, or torch tensor.
        y_pred: Predicted probabilities. Can be list, numpy array, or torch tensor.

    Returns:
        float: The AUC score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    try:
        # Check if we have more than one class to calculate AUC
        if len(np.unique(y_true)) < 2:
            # Fallback if batch/subset only has one class (e.g. during debugging or small batches)
            # Returning 0.5 as a neutral score
            return 0.5

        score = roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5

    return score


def setup_logger(log_file_path: str = None):
    """
    Sets up a logger that writes to console and optionally to a file.

    Args:
        log_file_path (str, optional): Path to the log file.
                                       If None, creates 'training.log' in Config.WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file_path is None:
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        log_file_path = os.path.join(Config.WORKING_DIR, "training.log")

    logger = logging.getLogger("Idea8_Logger")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if function is called repeatedly
    if not logger.handlers:
        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        # File Handler
        f_handler = logging.FileHandler(log_file_path)
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger
