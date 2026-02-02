import numpy as np


def log_transform(y):
    """
    Applies the log(1+x) transformation to the input data.
    Useful for targets with a wide range or when optimizing for RMSLE.

    Args:
        y (array-like): Input data.

    Returns:
        np.ndarray: Transformed data.
    """
    return np.log1p(y)


def inverse_log_transform(y):
    """
    Applies the exp(x)-1 transformation to reverse log_transform.

    Args:
        y (array-like): Log-transformed data.

    Returns:
        np.ndarray: Data in original scale.
    """
    return np.expm1(y)


def calculate_rmsle(y_true, y_pred):
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    The metric is computed by taking the RMSE of the log-transformed values for each
    column (target variable) independently, and then averaging these scores.

    Args:
        y_true (np.ndarray or pd.DataFrame): Ground truth values.
        y_pred (np.ndarray or pd.DataFrame): Predicted values.

    Returns:
        float: The mean RMSLE across all target columns.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to prevent domain errors with log (predictions cannot be negative for physical energies)
    # This handles potential negative predictions from linear models
    y_pred = np.maximum(y_pred, 0)

    # Calculate Log(1+x)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Calculate Squared Errors
    squared_errors = np.square(log_true - log_pred)

    # Calculate Mean Squared Error per column
    mse_per_col = np.mean(squared_errors, axis=0)

    # Calculate RMSE per column (RMSLE)
    rmsle_per_col = np.sqrt(mse_per_col)

    # Return the mean of the column-wise RMSLEs
    return np.mean(rmsle_per_col)
