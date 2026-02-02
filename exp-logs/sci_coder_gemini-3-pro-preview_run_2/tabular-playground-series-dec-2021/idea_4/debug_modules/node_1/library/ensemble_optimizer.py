import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def optimize_blending_weights(oof_dict, y_true):
    """
    Optimizes the scalar weights for blending multiple model OOF predictions
    to minimize the Multi-class Logarithmic Loss.

    Args:
        oof_dict (dict): A dictionary where keys are model identifiers (str)
                         and values are numpy arrays of shape (n_samples, n_classes)
                         representing the OOF probability predictions.
        y_true (np.ndarray): The true target class labels of shape (n_samples,).

    Returns:
        dict: A dictionary mapping model identifiers to their optimized weights.
    """
    # Extract model names to ensure consistent order
    model_names = list(oof_dict.keys())

    # Stack predictions into a single tensor for efficient computation
    # Shape: (n_models, n_samples, n_classes)
    preds_stack = np.array([oof_dict[name] for name in model_names])

    n_models = len(model_names)

    # Initial guess: Equal weights for all models
    initial_weights = np.ones(n_models) / n_models

    # Bounds: Each weight must be between 0 and 1
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    # Constraints: The sum of weights must equal 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    print(f"Starting weight optimization for {n_models} models...")

    # Define the objective function to minimize
    def objective(weights):
        # Reshape weights for broadcasting: (n_models, 1, 1)
        w_reshaped = weights[:, np.newaxis, np.newaxis]

        # Compute weighted average: sum(weight_i * pred_i)
        weighted_preds = np.sum(preds_stack * w_reshaped, axis=0)

        # Clip predictions to avoid log(0) errors (numerical stability)
        # Although sklearn log_loss handles this, explicit clipping is safer for custom loops
        weighted_preds = np.clip(weighted_preds, 1e-15, 1 - 1e-15)

        # Renormalize rows to ensure they sum to 1 exactly (fix float drift)
        weighted_preds = weighted_preds / weighted_preds.sum(axis=1, keepdims=True)

        return log_loss(y_true, weighted_preds)

    # Run the optimization
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False, "maxiter": 1000},
    )

    # Output results
    print(f"Optimization finished. Success: {result.success}")
    print(f"Final Optimized OOF LogLoss: {result.fun}")

    # Construct result dictionary
    optimized_weights = result.x
    weights_dict = {name: float(w) for name, w in zip(model_names, optimized_weights)}

    print("Optimized Weights:")
    for name, w in weights_dict.items():
        print(f"  {name}: {w}")

    return weights_dict


def weighted_average(preds_list, weights):
    """
    Computes the weighted average of a list of prediction arrays.

    Args:
        preds_list (list of np.ndarray): List of probability arrays, each of shape (n_samples, n_classes).
        weights (list of float): List of scalar weights corresponding to preds_list.

    Returns:
        np.ndarray: The weighted average prediction array of shape (n_samples, n_classes).
    """
    if len(preds_list) != len(weights):
        raise ValueError(
            f"Mismatch: {len(preds_list)} prediction arrays but {len(weights)} weights provided."
        )

    # Convert weights to numpy array and normalize to sum to 1
    w_arr = np.array(weights, dtype=float)
    w_sum = np.sum(w_arr)

    if w_sum == 0:
        print("Warning: Sum of weights is 0. Reverting to equal weights.")
        w_arr = np.ones(len(weights)) / len(weights)
    else:
        w_arr = w_arr / w_sum

    # Initialize result array
    weighted_preds = np.zeros_like(preds_list[0], dtype=float)

    # Accumulate weighted predictions
    for pred, w in zip(preds_list, w_arr):
        weighted_preds += pred * w

    return weighted_preds
