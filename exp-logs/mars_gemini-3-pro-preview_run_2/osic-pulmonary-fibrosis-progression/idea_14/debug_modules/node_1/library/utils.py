import numpy as np
from library.config import Config


def seed_everything(seed=None):
    """
    Sets random seeds for reproducibility across Python, Numpy, and Torch.
    Delegates to the Config class implementation to avoid code duplication.

    Args:
        seed (int, optional): Specific seed to use. If None, uses Config.SEED.
    """
    Config.set_seed(seed)


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric as defined in the task.

    The metric is defined as:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or list): True FVC values (ground truth).
        y_pred (np.array or list): Predicted FVC values.
        sigma (np.array or list): Predicted confidence values (standard deviation).

    Returns:
        float: The average metric score across all samples.
    """
    # Ensure inputs are numpy arrays of float type for vectorized operations
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    sigma = np.array(sigma, dtype=np.float64)

    # Clip sigma to reflect approximate measurement uncertainty (min 70 ml)
    sigma_clipped = np.maximum(sigma, 70)

    # Calculate absolute error (delta), thresholded at 1000 ml to avoid excessive penalties
    delta = np.minimum(np.abs(y_true - y_pred), 1000)

    # Compute the metric formula
    # metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    # Return the mean score
    return np.mean(metric_values)
