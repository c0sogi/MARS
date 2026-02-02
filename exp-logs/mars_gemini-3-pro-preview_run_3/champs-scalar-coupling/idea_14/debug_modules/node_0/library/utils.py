import numpy as np
import pandas as pd
import torch
import random
import os
from library.config import Config


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


def calculate_log_mae(y_true, y_pred, types):
    """
    Calculates the Log of the Mean Absolute Error, calculated for each scalar coupling type,
    and then averaged across types. This is the official competition metric.

    Args:
        y_true (np.ndarray or pd.Series): True target values.
        y_pred (np.ndarray or pd.Series): Predicted target values.
        types (np.ndarray or pd.Series): Coupling types corresponding to the values.

    Returns:
        float: The Log Mean Absolute Error score.
    """
    # Ensure inputs are numpy arrays for consistent processing
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(types, pd.Series):
        types = types.values

    # Create a DataFrame to leverage groupby functionality
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "type": types})

    # Calculate absolute error for each prediction
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])

    # Calculate Mean Absolute Error for each coupling type
    mae_per_type = df.groupby("type")["abs_error"].mean()

    # Take the natural logarithm of the MAE for each type
    # We add a small epsilon inside log if necessary, but competition metric implies log(MAE)
    # Assuming MAE is non-zero. If MAE is 0, log is -inf.
    # In practice, MAE is rarely exactly 0.
    log_mae_per_type = np.log(mae_per_type)

    # Average the log MAEs across all types
    score = log_mae_per_type.mean()

    return score


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across numpy, torch, and python random.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
