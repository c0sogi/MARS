import os
import sys
import random
import logging
import copy
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Also configures CUDNN backend settings based on Config.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configure CUDNN based on Config
    # CUDNN_BENCHMARK = True is recommended for fixed input sizes (32x32)
    if Config.CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
        # When benchmark is True, deterministic algorithms are often not available
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def setup_logger(name="cactus_logger", log_file=None):
    """
    Sets up a logger to print to console and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if function is called repeatedly
    if not logger.handlers:
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler (optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve.

    Args:
        y_true: Ground truth labels (numpy array or list).
        y_pred: Predicted probabilities (numpy array or list).

    Returns:
        float: The AUC score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check if we have both classes to calculate AUC
    if len(np.unique(y_true)) < 2:
        # Fallback if batch/fold has only one class (rare in stratified splits but possible in small debug)
        return 0.5

    return roc_auc_score(y_true, y_pred)


def save_checkpoint(model, path):
    """
    Safely saves the model state dictionary using deepcopy to ensure immutability.

    Args:
        model: The PyTorch model to save.
        path: The file path to save the checkpoint to.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Deepcopy the state dict to ensure we have an immutable version of the weights
    # at this exact moment, preventing issues if the model continues training or sharing memory.
    state_dict = copy.deepcopy(model.state_dict())

    torch.save(state_dict, path)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
