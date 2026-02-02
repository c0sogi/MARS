import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa score.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # If predictions are float, round them to nearest integer for safety
    # (Though usually this function is called with integer predictions)
    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = np.rint(y_pred).astype(int)

    # Ensure y_true are integers
    if np.issubdtype(y_true.dtype, np.floating):
        y_true = np.rint(y_true).astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def apply_thresholds(y_pred, thresholds):
    """
    Applies a set of thresholds to continuous predictions to generate integer class labels (1-6).

    Args:
        y_pred (array-like): Continuous regression predictions.
        thresholds (array-like): A list or array of 5 threshold values.

    Returns:
        np.ndarray: Integer class labels.
    """
    y_pred = np.array(y_pred)
    thresholds = np.sort(thresholds)

    # Initialize all predictions to the lowest class (1)
    y_pred_int = np.full_like(y_pred, 1, dtype=int)

    # For each threshold, if the prediction is higher, increment the class
    # Logic:
    # x < t0 -> 1
    # t0 <= x < t1 -> 2
    # ...
    # t4 <= x -> 6
    for i, t in enumerate(thresholds):
        y_pred_int[y_pred > t] = i + 2

    return y_pred_int


def optimize_thresholds(y_true, y_pred_continuous, init_thresholds=None):
    """
    Finds the optimal decision boundaries (thresholds) that maximize the QWK score
    using the Nelder-Mead optimization algorithm.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred_continuous (array-like): Continuous regression predictions.
        init_thresholds (list, optional): Initial guess for thresholds.
                                          Defaults to [1.5, 2.5, 3.5, 4.5, 5.5].

    Returns:
        np.ndarray: The optimized thresholds.
    """
    if init_thresholds is None:
        # Default midpoints for a 1-6 scale
        init_thresholds = [1.5, 2.5, 3.5, 4.5, 5.5]

    def objective(thresholds):
        # 1. Apply current thresholds to get integer labels
        preds_int = apply_thresholds(y_pred_continuous, thresholds)
        # 2. Calculate QWK
        score = compute_qwk(y_true, preds_int)
        # 3. Return negative score because we want to maximize QWK
        return -score

    # Run optimization
    result = minimize(
        objective, init_thresholds, method="Nelder-Mead", options={"maxiter": 500}
    )

    # Return sorted thresholds to ensure logical consistency
    return np.sort(result.x)
