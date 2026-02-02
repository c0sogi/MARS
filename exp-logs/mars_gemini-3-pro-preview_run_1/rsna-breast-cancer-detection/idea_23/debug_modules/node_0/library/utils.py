import os
import sys
import random
import numpy as np
import torch
import logging


def set_seed(seed=42):
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducible results.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name, log_file=None):
    """
    Configures and returns a logger that writes to console and optionally a file.

    Args:
        name (str): The name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class ProbabilisticF1:
    """
    Computes the Probabilistic F1 score (pF1) for binary classification.

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Where:
        pTP = sum(y_pred * y_true)
        pFP = sum(y_pred * (1 - y_true))
        TP + FN = sum(y_true) (Total actual positives)
    """

    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon

    def __call__(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor or np.ndarray): Predicted probabilities (0 to 1).
            y_true (torch.Tensor or np.ndarray): Ground truth labels (0 or 1).

        Returns:
            float: The probabilistic F1 score.
        """
        # Convert tensors to numpy if necessary
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()

        # Flatten arrays to ensure 1D
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # Calculate probabilistic True Positives (pTP)
        # pTP is the sum of probabilities for positive samples
        p_tp = np.sum(y_pred * y_true)

        # Calculate probabilistic False Positives (pFP)
        # pFP is the sum of probabilities for negative samples
        p_fp = np.sum(y_pred * (1 - y_true))

        # Calculate Total Positives (TP + FN)
        # This is simply the count of positive ground truth labels
        total_positives = np.sum(y_true)

        # Calculate pPrecision
        # Denominator is sum of all predicted probabilities (pTP + pFP)
        denominator_prec = p_tp + p_fp
        p_precision = p_tp / (denominator_prec + self.epsilon)

        # Calculate pRecall
        # Denominator is total actual positives
        p_recall = p_tp / (total_positives + self.epsilon)

        # Calculate pF1
        denominator_f1 = p_precision + p_recall
        p_f1 = 2 * (p_precision * p_recall) / (denominator_f1 + self.epsilon)

        return float(p_f1)


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
