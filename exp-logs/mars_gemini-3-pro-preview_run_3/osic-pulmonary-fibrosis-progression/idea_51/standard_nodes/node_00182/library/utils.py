import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
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


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training loops.
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


def laplace_log_likelihood_score(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric as defined in the task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: True FVC values (numpy array or scalar).
        y_pred: Predicted FVC values (numpy array or scalar).
        sigma: Predicted confidence (std dev) values (numpy array or scalar).

    Returns:
        float: The average metric score (higher is better, values are typically negative).
    """
    # Ensure inputs are numpy arrays for element-wise operations
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Retrieve constants from Config
    sigma_min = Config.SIGMA_MIN
    max_error = Config.MAX_ERROR
    sqrt_2 = Config.SQRT_2

    # 1. Clip sigma (confidence) at 70 ml
    sigma_clipped = np.maximum(sigma, sigma_min)

    # 2. Calculate absolute error and clip at 1000 ml
    abs_diff = np.abs(y_true - y_pred)
    delta = np.minimum(abs_diff, max_error)

    # 3. Compute metric
    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = np.log(sqrt_2 * sigma_clipped)
    metric = -term1 - term2

    return np.mean(metric)


def get_global_stats(csv_path=None):
    """
    Calculates the global mean and standard deviation of the FVC target
    from the training set. This is used for target normalization and
    architectural constraints.

    Args:
        csv_path (str, optional): Path to the training CSV file.
                                  If None, uses Config.TRAIN_CSV.

    Returns:
        tuple: (mean_fvc, std_fvc)
    """
    if csv_path is None:
        csv_path = Config.TRAIN_CSV

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if "FVC" not in df.columns:
        raise ValueError("Column 'FVC' not found in the provided CSV.")

    mean_fvc = df["FVC"].mean()
    std_fvc = df["FVC"].std()

    return mean_fvc, std_fvc
