import os
import sys
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    Args:
        y_true (np.array or torch.Tensor): Ground truth binary labels (0 or 1).
        y_pred (np.array or torch.Tensor): Predicted probabilities (0 to 1).
        epsilon (float): Small value to prevent division by zero.

    Returns:
        float: The pF1 score.
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    # Calculate Probabilistic True Positives (pTP)
    # pTP = Sum(y_true * y_pred)
    p_tp = np.sum(y_true * y_pred)

    # Calculate Probabilistic Precision
    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP = Sum(y_true * y_pred) + Sum((1 - y_true) * y_pred) = Sum(y_pred)
    sum_pred = np.sum(y_pred)
    p_precision = p_tp / (sum_pred + epsilon)

    # Calculate Probabilistic Recall
    # pRecall = pTP / (TP + FN)
    # Note: TP + FN is simply the count of positive ground truth labels
    total_positives = np.sum(y_true)
    p_recall = p_tp / (total_positives + epsilon)

    # Calculate pF1
    p_f1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return p_f1


class Logger:
    """
    A simple logger to write output to both the console and a log file.
    """

    def __init__(self, log_file_name="train_log.txt"):
        self.log_path = os.path.join(Config.WORKING_DIR, log_file_name)

        # Initialize file (overwrite if exists)
        with open(self.log_path, "w") as f:
            f.write(f"Log started for experiment in {Config.WORKING_DIR}\n")

    def log(self, message):
        """
        Prints message to stdout and appends to log file.
        """
        # Print to console
        print(message)

        # Write to file
        with open(self.log_path, "a") as f:
            f.write(str(message) + "\n")

    def log_metrics(self, epoch, train_loss, val_loss, val_pf1):
        """
        Logs training metrics with full precision.
        """
        message = (
            f"Epoch {epoch}: "
            f"Train Loss = {train_loss}, "
            f"Val Loss = {val_loss}, "
            f"Val pF1 = {val_pf1}"
        )
        self.log(message)
