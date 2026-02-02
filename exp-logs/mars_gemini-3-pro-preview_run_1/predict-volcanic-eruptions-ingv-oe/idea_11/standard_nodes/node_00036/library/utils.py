import os
import random
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from library.config import SEED


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed to use. Defaults to the value in config.py.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_mae(y_true, y_pred):
    """
    Calculates the Mean Absolute Error between true and predicted values.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.

    Returns:
        float: The Mean Absolute Error.
    """
    return mean_absolute_error(y_true, y_pred)


def log_transform_target(y):
    """
    Applies a log transformation (log(1+x)) to the target variable.
    This is used to handle the wide range of time_to_eruption values and
    stabilize the training of the Vision branch.

    Args:
        y (array-like or float): The target value(s).

    Returns:
        array-like or float: The log-transformed target(s).
    """
    return np.log1p(y)


def exp_transform_target(y):
    """
    Applies an exponential transformation (exp(x) - 1) to the target variable.
    This is the inverse of log_transform_target, used to convert model predictions
    back to the original time scale.

    Args:
        y (array-like or float): The log-transformed target value(s).

    Returns:
        array-like or float: The original scale target(s).
    """
    # Clamp to avoid overflow in expm1 (float32 max is approx exp(88.7))
    # Real targets are around exp(18), so 85 is a safe upper bound
    y = np.clip(y, None, 85.0)
    return np.expm1(y)
