import numpy as np
import torch


def to_numpy(data):
    """
    Helper function to convert input data to a numpy array.
    Handles lists, numpy arrays, and torch Tensors (on CPU or GPU).
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def log_transform(y):
    """
    Applies log(1+x) transformation to the input data.
    This is used to transform the targets (formation energy and bandgap)
    to align the regression loss (MSE) with the RMSLE metric.

    Args:
        y: Input data (numpy array or torch Tensor).

    Returns:
        Transformed data (same type as input).
    """
    if isinstance(y, torch.Tensor):
        return torch.log1p(y)
    return np.log1p(y)


def inverse_log_transform(y_pred):
    """
    Applies exp(x) - 1 transformation to reverse the log_transform.
    Used to convert model predictions back to the original energy scale.

    Args:
        y_pred: Log-transformed predictions (numpy array or torch Tensor).

    Returns:
        Data in original scale (same type as input).
    """
    if isinstance(y_pred, torch.Tensor):
        return torch.expm1(y_pred)
    return np.expm1(y_pred)


def compute_rmsle(y_true, y_pred):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    The metric is defined as the RMSE of the log-transformed values.
    Predictions are clipped to be non-negative to avoid domain errors in log.

    Args:
        y_true: Ground truth values in original scale.
        y_pred: Predicted values in original scale.

    Returns:
        float: The mean RMSLE averaged across all target columns.
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    # Clip predictions to non-negative range to avoid domain errors in log
    # Physical energies (formation and bandgap) in this dataset are non-negative
    y_pred = np.maximum(y_pred, 0)

    # Calculate log(1+x)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Compute squared errors
    squared_log_errors = np.square(log_pred - log_true)

    # Compute Mean Squared Log Error for each column (target variable)
    # axis=0 averages over the batch dimension
    msle_per_column = np.mean(squared_log_errors, axis=0)

    # Compute Root Mean Squared Log Error for each column
    rmsle_per_column = np.sqrt(msle_per_column)

    # Return the average RMSLE across columns
    return np.mean(rmsle_per_column)
