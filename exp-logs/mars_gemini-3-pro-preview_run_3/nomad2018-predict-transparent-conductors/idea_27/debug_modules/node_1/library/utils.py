import numpy as np


def log_transform(y):
    """
    Apply natural log transformation: z = log(1 + y).
    Used for target variables to stabilize variance and handle skewed distributions.
    """
    return np.log1p(y)


def inverse_log_transform(z):
    """
    Apply inverse log transformation: y = exp(z) - 1.
    Used to convert model predictions back to the original energy scale.
    """
    return np.expm1(z)


def safe_mean(arr):
    """
    Compute the arithmetic mean of an array.
    Returns np.nan if the array is empty, ensuring downstream models can handle missing data.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.nan
    return np.mean(arr)


def safe_variance(arr):
    """
    Compute the variance of an array.
    Returns 0.0 if the array is empty or has only one element (no variation).
    """
    arr = np.asarray(arr)
    if arr.size <= 1:
        return 0.0
    return np.var(arr)


def safe_percentile(arr, q):
    """
    Compute the q-th percentile(s) of the data.
    Handles empty arrays by returning np.nan (or an array of NaNs if q is a sequence).

    Args:
        arr: Input data.
        q: Percentile or sequence of percentiles to compute (0-100).
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        # If q is a list/array, return an array of NaNs of the same shape
        if np.ndim(q) > 0:
            return np.full(len(q), np.nan)
        else:
            return np.nan
    return np.percentile(arr, q)


def safe_min(arr):
    """
    Return the minimum of an array.
    Returns np.nan if the array is empty.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.nan
    return np.min(arr)


def safe_max(arr):
    """
    Return the maximum of an array.
    Returns np.nan if the array is empty.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.nan
    return np.max(arr)
