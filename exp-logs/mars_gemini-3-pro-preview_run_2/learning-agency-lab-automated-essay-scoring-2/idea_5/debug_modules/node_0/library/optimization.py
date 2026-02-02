import numpy as np
from scipy.optimize import minimize
from library.utils import compute_qwk


def apply_thresholds(y_pred, thresholds):
    """
    Applies thresholds to continuous predictions to obtain integer class labels (1-6).

    Args:
        y_pred (np.ndarray): Continuous regression predictions.
        thresholds (np.ndarray): Array of 5 thresholds.

    Returns:
        np.ndarray: Integer predictions in range [1, 6].
    """
    # Ensure thresholds are sorted to maintain monotonic logic
    thresholds = np.sort(thresholds)

    # np.digitize returns indices of the bins to which each value belongs.
    # bins[i-1] <= x < bins[i]
    # With 5 thresholds [t0, t1, t2, t3, t4]:
    # x < t0 -> index 0 -> Class 1
    # t0 <= x < t1 -> index 1 -> Class 2
    # ...
    # x >= t4 -> index 5 -> Class 6
    return np.digitize(y_pred, thresholds) + 1


def optimize_thresholds(y_true, y_pred, initial_thresholds=None, max_iter=500):
    """
    Finds the optimal thresholds that maximize the Quadratic Weighted Kappa score
    on the provided data using the Nelder-Mead method.

    Args:
        y_true (np.ndarray): True integer labels.
        y_pred (np.ndarray): Continuous regression predictions.
        initial_thresholds (list or np.ndarray, optional): Starting thresholds.
                                                           Defaults to [1.5, 2.5, 3.5, 4.5, 5.5].
        max_iter (int): Maximum number of iterations for the optimizer.

    Returns:
        np.ndarray: The optimized, sorted thresholds.
    """
    if initial_thresholds is None:
        # Standard rounding boundaries for 1-6 scale
        initial_thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    else:
        initial_thresholds = np.array(initial_thresholds)

    # Objective function to minimize (negative QWK)
    def negative_qwk(thresholds):
        # Apply current thresholds
        y_pred_int = apply_thresholds(y_pred, thresholds)
        # Compute QWK
        score = compute_qwk(y_true, y_pred_int)
        # Return negative for minimization
        return -score

    # Run optimization
    # Nelder-Mead is a heuristic search method that doesn't require gradients,
    # making it suitable for optimizing thresholds for a non-differentiable metric like QWK.
    result = minimize(
        negative_qwk,
        initial_thresholds,
        method="Nelder-Mead",
        options={"maxiter": max_iter},
    )

    # Return sorted optimized thresholds
    return np.sort(result.x)
