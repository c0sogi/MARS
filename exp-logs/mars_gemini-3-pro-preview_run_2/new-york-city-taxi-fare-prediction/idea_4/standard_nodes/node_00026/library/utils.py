import numpy as np


def log_transform(y):
    """
    Apply log(1 + y) transformation to the target variable.
    This helps in handling heavy-tailed distributions and stabilizing
    gradients during training.

    Args:
        y (np.ndarray or float): Original target values (fare_amount).

    Returns:
        np.ndarray or float: Log-transformed values.
    """
    return np.log1p(y)


def inverse_log_transform(y_log):
    """
    Apply exp(y) - 1 transformation to revert log-transformed values.
    This is used to convert model predictions back to the original dollar scale.

    Args:
        y_log (np.ndarray or float): Log-transformed values.

    Returns:
        np.ndarray or float: Original scale values.
    """
    return np.expm1(y_log)


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The RMSE value.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
