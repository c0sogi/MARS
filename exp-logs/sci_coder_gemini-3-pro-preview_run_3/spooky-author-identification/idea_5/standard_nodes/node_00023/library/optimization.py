import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from library.config import Config


def _calculate_weighted_log_loss(weights, oof_preds_list, y_true):
    """
    Objective function for optimization.
    Calculates the log loss of the weighted ensemble.

    Args:
        weights (list or np.array): The weights for each model.
        oof_preds_list (list of np.array): List of OOF probability matrices.
        y_true (np.array): True class indices.

    Returns:
        float: The log loss score.
    """
    # Initialize final predictions
    final_preds = np.zeros_like(oof_preds_list[0])

    # Weighted sum
    for i, pred in enumerate(oof_preds_list):
        final_preds += weights[i] * pred

    # Metric Implementation Details from Task Description:
    # 1. Rescale: each row is divided by the row sum
    row_sums = np.sum(final_preds, axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1e-15
    final_preds = final_preds / row_sums[:, np.newaxis]

    # 2. Clip: max(min(p, 1-10^-15), 10^-15)
    final_preds = np.clip(final_preds, 1e-15, 1 - 1e-15)

    # Calculate Log Loss
    score = log_loss(y_true, final_preds)
    return score


def optimize_weights(oof_dict, y_true):
    """
    Finds the optimal weights for blending multiple models using SLSQP.

    Args:
        oof_dict (dict): Dictionary where keys are model names and values are
                         OOF prediction arrays of shape (n_samples, n_classes).
        y_true (np.array): Array of true class labels (integers).

    Returns:
        dict: A dictionary mapping model names to their optimized weights.
    """
    model_names = list(oof_dict.keys())
    oof_preds_list = [oof_dict[name] for name in model_names]
    n_models = len(model_names)

    # Initial guess: Equal weights
    initial_weights = np.ones(n_models) / n_models

    # Constraints: Weights must sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Bounds: Weights must be between 0 and 1
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    print(f"Optimizing ensemble weights for {n_models} models: {model_names}...")

    # Run optimization
    result = minimize(
        fun=_calculate_weighted_log_loss,
        x0=initial_weights,
        args=(oof_preds_list, y_true),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=1e-6,
    )

    optimized_weights = result.x
    final_score = result.fun

    print(f"Optimization finished. Success: {result.success}")
    print(f"Optimized Ensemble Log Loss: {final_score}")

    weight_dict = {}
    print("Final Weights:")
    for name, w in zip(model_names, optimized_weights):
        print(f"  {name}: {w:.6f}")
        weight_dict[name] = w

    return weight_dict


def apply_ensemble(preds_dict, weights_dict):
    """
    Applies the optimized weights to a dictionary of predictions.

    Args:
        preds_dict (dict): Dictionary of prediction arrays (e.g., test set predictions).
        weights_dict (dict): Dictionary of weights (output of optimize_weights).

    Returns:
        np.array: The final weighted, normalized, and clipped probabilities.
    """
    model_names = list(preds_dict.keys())

    # Use the first prediction array to initialize the shape
    first_pred = preds_dict[model_names[0]]
    final_preds = np.zeros_like(first_pred)

    total_weight_applied = 0.0

    for name in model_names:
        if name in weights_dict:
            w = weights_dict[name]
            final_preds += w * preds_dict[name]
            total_weight_applied += w
        else:
            # If a model in preds is not in weights (shouldn't happen in standard flow), ignore or warn
            pass

    # If total weight is effectively zero (e.g. empty dict), avoid issues
    if total_weight_applied == 0:
        return np.ones_like(final_preds) / final_preds.shape[1]

    # Apply Metric specific post-processing

    # 1. Rescale (Row normalization)
    row_sums = np.sum(final_preds, axis=1)
    row_sums[row_sums == 0] = 1e-15
    final_preds = final_preds / row_sums[:, np.newaxis]

    # 2. Clip
    final_preds = np.clip(final_preds, 1e-15, 1 - 1e-15)

    return final_preds
