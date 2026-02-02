import numpy as np
from library.config import Config


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric for the Lung Function Decline Prediction task.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.ndarray or list): Ground truth FVC values.
        y_pred (np.ndarray or list): Predicted FVC values.
        sigma (np.ndarray or list): Predicted confidence values (standard deviation).

    Returns:
        float: The average metric score across all samples.
    """
    # Ensure inputs are numpy arrays of float type
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Clip confidence values to the minimum threshold (default 70)
    sigma_clipped = np.maximum(sigma, Config.MIN_CONFIDENCE)

    # Calculate absolute error
    abs_error = np.abs(y_true - y_pred)

    # Threshold the error to avoid large penalties (default 1000)
    delta = np.minimum(abs_error, Config.MAX_ERROR_THRESHOLD)

    # Compute the metric term by term
    # Term 1: - (sqrt(2) * delta / sigma_clipped)
    term1 = -(np.sqrt(2) * delta) / sigma_clipped

    # Term 2: - ln(sqrt(2) * sigma_clipped)
    term2 = -np.log(np.sqrt(2) * sigma_clipped)

    # Combine terms
    metric = term1 + term2

    # Return the mean score across the batch
    return np.mean(metric)
