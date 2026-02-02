import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across libraries.
    Delegates to the Config.seed_everything method.

    Args:
        seed (int): The seed value to set.
    """
    Config.seed_everything(seed)


def log1p_transform(y):
    """
    Applies log(1+x) transformation to the input data.
    Useful for stabilizing targets with wide ranges or skew.

    Args:
        y: Input array or tensor.

    Returns:
        Transformed data (same type as input).
    """
    if hasattr(y, "log1p"):
        return y.log1p()
    return np.log1p(y)


def expm1_transform(y):
    """
    Applies exp(x)-1 inverse transformation.
    Used to convert model predictions back to original scale.

    Args:
        y: Input array or tensor.

    Returns:
        Transformed data (same type as input).
    """
    if hasattr(y, "expm1"):
        return y.expm1()
    return np.expm1(y)


def rmsle(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Metric Definition:
        1. Compute log(1 + x) for both true and predicted values.
        2. Compute squared difference.
        3. Compute mean over samples (MSE of logs) for each column.
        4. Compute sqrt (RMSE of logs) for each column.
        5. Compute mean of these column-wise scores.

    Args:
        y_true: Ground truth values (numpy array or torch tensor).
        y_pred: Predicted values (numpy array or torch tensor).

    Returns:
        float: The mean of the RMSLE calculated for each column.
    """
    # Convert torch tensors to numpy arrays if necessary
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to be non-negative to avoid log domain errors (log of negative)
    # Physical energies should generally be handled carefully, but for log1p metric
    # inputs must be >= -1. We clip at 0 for safety as energies are typically positive
    # or the metric assumes non-negative inputs.
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Calculate squared logarithmic errors
    log_diff = np.log1p(y_true) - np.log1p(y_pred)
    squared_log_errors = log_diff**2

    # Calculate Mean Squared Logarithmic Error (MSLE) for each column (axis=0)
    msle_per_column = np.mean(squared_log_errors, axis=0)

    # Calculate RMSLE for each column
    rmsle_per_column = np.sqrt(msle_per_column)

    # Return the mean of column-wise RMSLEs
    return float(np.mean(rmsle_per_column))
