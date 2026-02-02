import numpy as np


def log_transform(y):
    """
    Applies the natural logarithm transformation z = log(1 + y) to the input.
    This is used to normalize the target distribution and match the evaluation metric.

    Args:
        y (np.ndarray or float): Input values (original scale).

    Returns:
        np.ndarray or float: Log-transformed values.
    """
    return np.log1p(y)


def inverse_log_transform(z):
    """
    Applies the exponential transformation y = exp(z) - 1 to the input.
    This is used to convert model predictions back to the original scale.

    Args:
        z (np.ndarray or float): Log-transformed values.

    Returns:
        np.ndarray or float: Values in original scale.
    """
    return np.expm1(z)


def calculate_rmsle(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    The metric is defined as the mean of the RMSLE calculated for each target column separately.
    It handles negative predictions by clipping them to 0 before applying the log transform.

    Args:
        y_true (np.ndarray): Ground truth values in original scale.
        y_pred (np.ndarray): Predicted values in original scale.

    Returns:
        float: The mean RMSLE across all columns.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure predictions are non-negative as log is undefined for negative numbers
    # Physical energies like bandgap should be non-negative.
    y_pred = np.maximum(y_pred, 0)

    # Apply log1p transformation: log(1 + x)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Calculate Squared Logarithmic Errors
    squared_log_errors = (log_true - log_pred) ** 2

    # Calculate Mean Squared Logarithmic Error for each column
    # If input is 1D (n_samples,), axis=0 returns a scalar.
    # If input is 2D (n_samples, n_targets), axis=0 returns shape (n_targets,).
    msle_per_column = np.mean(squared_log_errors, axis=0)

    # Calculate RMSLE for each column
    rmsle_per_column = np.sqrt(msle_per_column)

    # Return the average RMSLE across columns
    return np.mean(rmsle_per_column)
