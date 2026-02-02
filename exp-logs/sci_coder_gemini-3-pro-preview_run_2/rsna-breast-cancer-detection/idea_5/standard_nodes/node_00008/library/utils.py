import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "main", log_file: str = "train.log") -> logging.Logger:
    """
    Configures and returns a logger that writes to both stdout and a file.

    Args:
        name (str): The name of the logger.
        log_file (str): The name of the log file to be saved in Config.WORKING_DIR.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is already configured
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream Handler (Stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handler
        log_path = os.path.join(Config.WORKING_DIR, log_file)
        # Ensure directory exists (Config should handle this, but for safety)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ProbabilisticF1:
    """
    Accumulates predictions and targets to calculate the Probabilistic F1 score (pF1).

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    pPrecision = pTP / (pTP + pFP)
    pRecall = pTP / (TP + FN)

    Where:
    pTP = Sum(preds * targets)
    pFP = Sum(preds * (1 - targets))
    TP + FN = Sum(targets)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self.sum_p_tp = 0.0  # Probabilistic True Positives
        self.sum_preds = 0.0  # Sum of all predicted probabilities (pTP + pFP)
        self.sum_targets = 0.0  # Sum of all ground truth positives (TP + FN)
        self.count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predicted probabilities (0-1), shape (N,).
            targets (torch.Tensor): Ground truth labels (0 or 1), shape (N,).
        """
        # Ensure inputs are flat and on CPU
        preds = preds.detach().cpu().view(-1).float()
        targets = targets.detach().cpu().view(-1).float()

        # pTP = sum(y_hat * y)
        self.sum_p_tp += torch.sum(preds * targets).item()

        # Denominator for Precision: sum(y_hat) = pTP + pFP
        self.sum_preds += torch.sum(preds).item()

        # Denominator for Recall: sum(y) = TP + FN (Actual Positives)
        self.sum_targets += torch.sum(targets).item()

        self.count += preds.numel()

    def compute(self) -> float:
        """
        Computes the final pF1 score based on accumulated data.

        Returns:
            float: The pF1 score.
        """
        # Avoid division by zero
        if self.sum_preds == 0:
            p_precision = 0.0
        else:
            p_precision = self.sum_p_tp / self.sum_preds

        if self.sum_targets == 0:
            p_recall = 0.0
        else:
            p_recall = self.sum_p_tp / self.sum_targets

        if (p_precision + p_recall) == 0:
            pf1 = 0.0
        else:
            pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall)

        return pf1
