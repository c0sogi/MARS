import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
from library.config import seed_everything, Config


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true (array-like): Ground truth integer labels.
        y_pred (array-like): Predicted integer labels.

    Returns:
        float: The QWK score.
    """
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def apply_thresholds(y_pred_continuous, thresholds):
    """
    Applies a set of thresholds to continuous predictions to convert them
    into discrete integer scores (1-6).

    Args:
        y_pred_continuous (array-like): Continuous regression outputs.
        thresholds (array-like): A list or array of 5 threshold values.

    Returns:
        np.ndarray: Array of integer scores between 1 and 6.
    """
    # Ensure thresholds are sorted for np.digitize
    thresholds = np.sort(thresholds)

    # np.digitize returns the index of the bin each value belongs to.
    # Bins are defined by thresholds.
    # Example: thresholds = [1.5, 2.5, ...]
    # value < 1.5 -> index 0 -> Score 1
    # 1.5 <= value < 2.5 -> index 1 -> Score 2
    # ...
    # value >= 5.5 -> index 5 -> Score 6
    idxs = np.digitize(y_pred_continuous, thresholds)

    # Map indices (0-5) to scores (1-6)
    return idxs + 1


def optimize_thresholds(y_true, y_pred_continuous):
    """
    Finds the optimal thresholds that maximize the Quadratic Weighted Kappa
    between the true labels and the discretized predictions.

    Args:
        y_true (array-like): Ground truth integer labels (1-6).
        y_pred_continuous (array-like): Continuous regression predictions.

    Returns:
        np.ndarray: The optimized array of 5 threshold values.
    """
    # Initial guess: standard rounding boundaries (1.5, 2.5, 3.5, 4.5, 5.5)
    initial_thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    def objective(thresholds):
        # Apply current thresholds to get discrete predictions
        preds_discrete = apply_thresholds(y_pred_continuous, thresholds)

        # Calculate QWK
        score = compute_qwk(y_true, preds_discrete)

        # We want to maximize QWK, so minimize negative QWK
        return -score

    # Use Nelder-Mead as it is a direct search method (derivative-free)
    # and handles the non-differentiable nature of the discretization step well.
    result = minimize(
        objective,
        initial_thresholds,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-5},
    )

    # Return the optimized thresholds sorted
    return np.sort(result.x)
