import os
import random
import numpy as np
import torch
import pandas as pd
from scipy.stats import spearmanr
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise Spearman's Correlation Coefficient.

    Args:
        y_true: numpy array or pandas DataFrame of shape (n_samples, n_targets)
        y_pred: numpy array or pandas DataFrame of shape (n_samples, n_targets)

    Returns:
        float: The mean Spearman correlation score.
    """
    # Convert pandas DataFrames to numpy arrays if necessary
    if hasattr(y_true, "values"):
        y_true = y_true.values
    if hasattr(y_pred, "values"):
        y_pred = y_pred.values

    # Ensure shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    n_targets = y_true.shape[1]
    correlations = []

    for i in range(n_targets):
        # Extract the i-th column for true and predicted values
        t = y_true[:, i]
        p = y_pred[:, i]

        # Compute Spearman correlation
        # spearmanr returns a Result object (correlation, pvalue) or a tuple in older versions
        # We handle the case where inputs might be constant (leading to NaN)
        try:
            corr, _ = spearmanr(t, p)
        except Exception:
            corr = 0.0

        # Handle NaN result (e.g., if a column is constant)
        if np.isnan(corr):
            corr = 0.0

        correlations.append(corr)

    # Return the mean of the column-wise correlations
    return float(np.mean(correlations))


class TrainingLogger:
    """
    A simple logger to track training progress and print metrics with full precision.
    """

    def __init__(self):
        pass

    def log(self, message):
        """
        Prints a message to the console.
        """
        print(message)

    def log_epoch(self, epoch, train_loss, val_loss, val_score):
        """
        Logs the metrics for a specific epoch with full precision.

        Args:
            epoch (int): The current epoch number.
            train_loss (float): The training loss.
            val_loss (float): The validation loss.
            val_score (float): The validation metric score (Spearman correlation).
        """
        print(
            f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Score: {val_score}"
        )
