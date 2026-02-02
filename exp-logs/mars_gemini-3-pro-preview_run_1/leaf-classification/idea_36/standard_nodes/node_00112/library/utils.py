import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import softmax
from library.config import FLOAT_PRECISION


def alphanumeric_sort(columns):
    """
    Sorts a list of feature column names alphanumerically (lexicographically).

    This enforces the specific ordering (e.g., margin_1, margin_10, margin_11...)
    required to match the memory layout of the high-performance baseline and
    prevent floating-point associativity noise during matrix operations.

    Args:
        columns (list of str): List of column names to sort.

    Returns:
        list of str: Sorted list of column names.
    """
    return sorted(columns)


def stable_softmax(logits):
    """
    Computes the softmax function using high-precision float64 arithmetic.

    Uses scipy.special.softmax which implements the log-sum-exp trick
    (subtracting max) for numerical stability.

    Args:
        logits (np.ndarray): Input logits matrix of shape (n_samples, n_classes).

    Returns:
        np.ndarray: Probability matrix of shape (n_samples, n_classes) in float64.
    """
    # Cast to float64 to ensure precision before computation
    logits_64 = np.array(logits, dtype=FLOAT_PRECISION)
    return softmax(logits_64, axis=1)


def cholesky_solve(covariance, target):
    """
    Solves the linear system A * X = B using Cholesky decomposition.

    This method is preferred over explicit inversion or SVD for Positive Definite
    matrices (like the shrunk covariance matrix) as it avoids spectral truncation
    and provides an exact solution in float64 precision.

    Args:
        covariance (np.ndarray): Symmetric Positive Definite covariance matrix (A).
        target (np.ndarray): The right-hand side vector or matrix (B).

    Returns:
        np.ndarray: The solution matrix X.
    """
    # Ensure inputs are float64
    cov_64 = np.array(covariance, dtype=FLOAT_PRECISION)
    target_64 = np.array(target, dtype=FLOAT_PRECISION)

    # Compute Cholesky decomposition A = L L^T
    # lower=True returns the lower triangular matrix L
    c, lower = cho_factor(cov_64, lower=True)

    # Solve the system using the factorization
    return cho_solve((c, lower), target_64)


def clip_probabilities(probs):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] to prevent infinite
    log loss penalties, as required by the competition metric.

    Args:
        probs (np.ndarray): Array of probabilities.

    Returns:
        np.ndarray: Clipped probabilities in float64.
    """
    probs_64 = np.array(probs, dtype=FLOAT_PRECISION)
    epsilon = 1e-15
    # The metric definition says max(min(p, 1-10^-15), 10^-15)
    return np.maximum(np.minimum(probs_64, 1.0 - epsilon), epsilon)
