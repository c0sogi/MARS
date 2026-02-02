import numpy as np


def log_transform(y):
    """
    Applies the natural logarithm transformation log(1 + y) to the target variable.
    This is useful for stabilizing variance and handling non-negative target distributions.

    Args:
        y (float, list, or np.ndarray): Original target values (e.g., formation energy, bandgap).

    Returns:
        np.ndarray: Transformed values z = log(1 + y).
    """
    return np.log1p(y)


def inverse_log_transform(z):
    """
    Applies the inverse transformation exp(z) - 1 to convert predictions back to the original scale.
    This ensures that the predicted energy values remain non-negative (or close to the original domain).

    Args:
        z (float, list, or np.ndarray): Log-transformed values.

    Returns:
        np.ndarray: Values in original scale y = exp(z) - 1.
    """
    return np.expm1(z)
