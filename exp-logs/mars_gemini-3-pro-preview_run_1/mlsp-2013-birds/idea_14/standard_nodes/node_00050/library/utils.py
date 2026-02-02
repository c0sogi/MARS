import os
import sys
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the computing device (GPU or CPU) based on Config.

    Returns:
        torch.device: The device object.
    """
    return torch.device(Config.DEVICE)


def get_logger(log_file=None):
    """
    Sets up a logger that writes to console and optionally to a file.

    Args:
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("BirdSpeciesClassifier")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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


def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC) for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth labels (binary).
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    auc_scores = []
    # Iterate over each class
    for i in range(y_true.shape[1]):
        try:
            # Check if class has both positive and negative samples
            if len(np.unique(y_true[:, i])) == 2:
                auc_scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        except ValueError:
            continue

    if len(auc_scores) == 0:
        return 0.0

    return np.mean(auc_scores)


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): Path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    torch.save(state, filepath)

    if is_best:
        dirname = os.path.dirname(filepath)
        best_path = os.path.join(dirname, "model_best.pth")
        torch.save(state, best_path)
