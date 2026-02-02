import numpy as np
from library.config import MIN_CONFIDENCE, MAX_ERROR


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric as defined in the task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|y_true - y_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (array-like): True FVC values.
        y_pred (array-like): Predicted FVC values.
        sigma (array-like): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score across all samples.
    """
    # Ensure inputs are numpy arrays of float type for precision
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Clip confidence values (sigma) to the minimum threshold (default 70)
    sigma_clipped = np.maximum(sigma, MIN_CONFIDENCE)

    # Calculate absolute error
    abs_error = np.abs(y_true - y_pred)

    # Threshold the error (Delta) to the maximum allowed error (default 1000)
    delta = np.minimum(abs_error, MAX_ERROR)

    # Calculate the metric components
    sqrt_2 = np.sqrt(2)
    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = np.log(sqrt_2 * sigma_clipped)

    # Combine terms according to the formula
    metric_values = -term1 - term2

    # Return the mean score
    return np.mean(metric_values)
