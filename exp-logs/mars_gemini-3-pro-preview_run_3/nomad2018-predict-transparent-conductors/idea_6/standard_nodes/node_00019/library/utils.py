import numpy as np
from sklearn.metrics import mean_squared_log_error


def log_transform(y):
    """
    Applies the log(1 + y) transformation to the target variables.
    This is used to normalize the target distribution and align with the RMSLE metric.

    Args:
        y (np.ndarray or pd.Series): Target values.

    Returns:
        np.ndarray: Log-transformed values.
    """
    return np.log1p(y)


def inverse_log_transform(z):
    """
    Applies the exp(z) - 1 transformation to reverse the log transform.
    Used to convert model predictions back to the original scale.

    Args:
        z (np.ndarray or pd.Series): Log-transformed values.

    Returns:
        np.ndarray: Values in original scale.
    """
    return np.expm1(z)


def calculate_rmsle(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    The metric is defined as the average of the RMSLE calculated for each target column independently.
    This function handles multi-output regression by iterating over columns.

    Args:
        y_true (np.ndarray): Ground truth values. Shape (n_samples, n_targets).
        y_pred (np.ndarray): Predicted values. Shape (n_samples, n_targets).

    Returns:
        float: The mean RMSLE score across all target columns.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predicted values to 0 to avoid errors with log(negative) and ensure validity
    # Physical quantities like energy/bandgap in this dataset are non-negative.
    y_pred = np.maximum(y_pred, 0)

    # Handle 1D arrays by reshaping to (n_samples, 1) for consistent processing
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_true.shape[1]
    rmsle_scores = []

    for i in range(n_targets):
        # Calculate MSLE for the current column
        # mean_squared_log_error calculates mean((log(1+y) - log(1+p))^2)
        try:
            msle = mean_squared_log_error(y_true[:, i], y_pred[:, i])
            rmsle = np.sqrt(msle)
            rmsle_scores.append(rmsle)
        except ValueError:
            # Fallback for empty or invalid columns, though unlikely with clipped preds
            rmsle_scores.append(0.0)

    # Return the average of the column-wise RMSLEs
    if not rmsle_scores:
        return 0.0

    return np.mean(rmsle_scores)
