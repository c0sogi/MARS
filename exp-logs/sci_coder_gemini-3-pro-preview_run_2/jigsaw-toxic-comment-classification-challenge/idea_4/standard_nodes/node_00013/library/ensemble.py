import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Union

from library.config import Config
from library.utils import calculate_roc_auc, save_submission


def get_val_targets() -> np.ndarray:
    """
    Loads the ground truth binary labels for the validation set.

    Returns:
        np.ndarray: Validation labels of shape (N_val, num_classes).
    """
    if not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError(f"Validation data not found at {Config.VAL_CSV}")

    df = pd.read_csv(Config.VAL_CSV)
    return df[Config.LABEL_COLS].values


def get_test_ids() -> List[str]:
    """
    Loads the IDs for the test set.

    Returns:
        List[str]: List of test IDs.
    """
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError(f"Test data not found at {Config.TEST_CSV}")

    df = pd.read_csv(Config.TEST_CSV)
    return df["id"].tolist()


def optimize_blending_weights(
    val_targets: np.ndarray, val_preds_list: List[np.ndarray], method: str = "SLSQP"
) -> np.ndarray:
    """
    Finds the optimal scalar weights for blending multiple model predictions
    to maximize the Mean Column-wise ROC AUC on the validation set.

    Args:
        val_targets: Ground truth labels (N, num_classes).
        val_preds_list: List of prediction arrays from different models, each (N, num_classes).
        method: Optimization algorithm for scipy.optimize.minimize.

    Returns:
        np.ndarray: Optimal weights summing to 1.0.
    """
    num_models = len(val_preds_list)
    if num_models == 0:
        raise ValueError("No predictions provided for optimization.")
    if num_models == 1:
        print("Only one model provided. Weight is 1.0.")
        return np.array([1.0])

    # Initial guess: Equal weights
    initial_weights = np.ones(num_models) / num_models

    # Constraints: Sum of weights must be 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Bounds: Weights must be between 0 and 1
    bounds = [(0.0, 1.0) for _ in range(num_models)]

    def objective(weights):
        # Normalize weights to ensure stability within the objective function
        # (though constraints should handle this, it prevents drift during steps)
        w_sum = np.sum(weights)
        if w_sum == 0:
            return 0.0
        w_norm = weights / w_sum

        # Compute blended predictions
        blended_preds = np.zeros_like(val_preds_list[0])
        for i, pred in enumerate(val_preds_list):
            blended_preds += w_norm[i] * pred

        # Calculate AUC
        # We want to maximize AUC, so we minimize negative AUC
        auc = calculate_roc_auc(val_targets, blended_preds)
        return -auc

    print(f"Optimizing ensemble weights for {num_models} models...")

    result = minimize(
        objective,
        initial_weights,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={"disp": False, "ftol": 1e-9},
    )

    if result.success:
        # Normalize final weights to be sure
        optimized_weights = result.x / np.sum(result.x)
        final_auc = -result.fun
        print(f"Optimization successful.")
        print(f"Best Validation AUC: {final_auc}")
        print(f"Optimal Weights: {optimized_weights}")
        return optimized_weights
    else:
        print(f"Optimization failed: {result.message}")
        print("Reverting to equal weights.")
        return initial_weights


def blend_predictions(preds_list: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """
    Computes the weighted average of a list of prediction arrays.

    Args:
        preds_list: List of prediction arrays (N, num_classes).
        weights: Array of scalar weights.

    Returns:
        np.ndarray: Blended predictions (N, num_classes).
    """
    if len(preds_list) != len(weights):
        raise ValueError(
            f"Mismatch: {len(preds_list)} prediction arrays vs {len(weights)} weights."
        )

    # Ensure weights sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    blended_preds = np.zeros_like(preds_list[0])
    for i, pred in enumerate(preds_list):
        blended_preds += weights[i] * pred

    return blended_preds


def create_submission(
    test_ids: List[str],
    test_preds: np.ndarray,
    output_path: str = Config.SUBMISSION_PATH,
) -> None:
    """
    Saves the final blended predictions to the submission CSV file.

    Args:
        test_ids: List of ID strings for the test set.
        test_preds: Blended probability predictions (N, num_classes).
        output_path: Path to save the CSV.
    """
    print(f"Saving submission file to {output_path}...")
    save_submission(test_ids, test_preds, output_path)
    print("Submission saved successfully.")
