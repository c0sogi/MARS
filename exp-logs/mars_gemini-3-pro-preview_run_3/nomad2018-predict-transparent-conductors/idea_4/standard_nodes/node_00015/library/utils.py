import numpy as np


def log_transform(y):
    """
    Applies the natural logarithm transformation log(1 + y).
    Useful for stabilizing variance and converting MSLE optimization to MSE.

    Args:
        y (np.ndarray or pd.Series): Target values.

    Returns:
        np.ndarray: Log-transformed values.
    """
    # Ensure input is a numpy array
    y = np.asarray(y)
    # Clip to avoid log of negative numbers (though data analysis shows min >= 0)
    y = np.maximum(y, 0)
    return np.log1p(y)


def inverse_log_transform(z):
    """
    Applies the inverse transformation exp(z) - 1.
    Used to convert model predictions back to the original scale.

    Args:
        z (np.ndarray or pd.Series): Log-scale predictions.

    Returns:
        np.ndarray: Original scale predictions.
    """
    z = np.asarray(z)
    return np.expm1(z)


def rmsle_score(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error (MCRMSE).

    Formula:
        RMSLE_col = sqrt( mean( (log1p(y_true_col) - log1p(y_pred_col))^2 ) )
        Score = mean( [RMSLE_col1, RMSLE_col2, ...] )

    Args:
        y_true (np.ndarray): Ground truth values (N_samples, N_targets).
        y_pred (np.ndarray): Predicted values (N_samples, N_targets).

    Returns:
        float: The mean column-wise RMSLE score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Ensure predictions are non-negative to avoid nan in log
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Calculate log(1 + y)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Squared Logarithmic Errors
    squared_log_errors = (log_true - log_pred) ** 2

    if y_true.ndim == 1:
        # Single target case
        return np.sqrt(np.mean(squared_log_errors))
    else:
        # Multi-target case: Calculate RMSE for each column first
        # mean over samples (axis 0)
        mse_per_column = np.mean(squared_log_errors, axis=0)
        rmsle_per_column = np.sqrt(mse_per_column)

        # Return the mean of the column-wise RMSLEs
        return np.mean(rmsle_per_column)
