import numpy as np
from scipy.optimize import minimize
from library.utils import calculate_metric
from library.config import Config


def optimize_weights(oof_preds_dict, y_true):
    """
    Optimizes the scalar weights for ensembling multiple probability streams
    to minimize Log Loss using constrained optimization (SLSQP).

    Args:
        oof_preds_dict (dict): Dictionary where keys are stream names (str)
                               and values are OOF probability matrices (np.ndarray).
        y_true (array-like): Ground truth class labels (integers).

    Returns:
        dict: Dictionary mapping stream names to their optimal scalar weights.
    """
    # Ensure y_true is a numpy array
    y_true = np.array(y_true)

    # Sort keys to ensure deterministic order of streams
    stream_names = sorted(list(oof_preds_dict.keys()))
    preds_list = [np.array(oof_preds_dict[k]) for k in stream_names]

    # Basic validation
    if not preds_list:
        raise ValueError("No predictions provided for optimization.")

    n_streams = len(preds_list)
    n_samples = len(y_true)

    # Verify shapes
    for i, preds in enumerate(preds_list):
        if len(preds) != n_samples:
            raise ValueError(
                f"Length mismatch: Stream '{stream_names[i]}' has {len(preds)} samples, y_true has {n_samples}."
            )

    # Initial weights: Uniform distribution
    initial_weights = np.array([1.0 / n_streams] * n_streams)

    # Define the objective function (Log Loss)
    def objective(weights):
        # Compute weighted average of probabilities
        # We initialize with zeros
        weighted_preds = np.zeros_like(preds_list[0])
        for i, w in enumerate(weights):
            weighted_preds += w * preds_list[i]

        # Calculate metric (handles clipping internally)
        return calculate_metric(y_true, weighted_preds)

    # Define Constraints: Sum of weights must be 1.0
    # scipy expects: fun(x) = 0
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Define Bounds: Each weight must be in [0, 1]
    bounds = [(0.0, 1.0) for _ in range(n_streams)]

    print(f"Starting ensemble weight optimization for {n_streams} streams...")

    # Run Optimization
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False, "ftol": 1e-9},
    )

    # Extract best weights
    best_weights = result.x

    # Explicit normalization to fix any minor numerical drift from optimization
    best_weights = best_weights / np.sum(best_weights)

    # Map weights back to stream names
    weight_dict = {name: w for name, w in zip(stream_names, best_weights)}
    final_loss = result.fun

    print("Optimization Complete.")
    print(f"Final Optimized Log Loss: {final_loss}")
    print("Optimal Weights:")
    for name in stream_names:
        print(f"  {name}: {weight_dict[name]:.6f}")

    return weight_dict


def apply_weights(preds_dict, weight_dict):
    """
    Applies the optimized weights to combine predictions from multiple streams.

    Args:
        preds_dict (dict): Dictionary of probability matrices (e.g., test predictions).
        weight_dict (dict): Dictionary of scalar weights.

    Returns:
        np.ndarray: The weighted ensemble probability matrix.
    """
    stream_names = list(weight_dict.keys())

    # Validation
    if not stream_names:
        raise ValueError("Weight dictionary is empty.")

    # Initialize accumulator
    # Use the shape of the first stream's predictions
    first_preds = preds_dict[stream_names[0]]
    final_preds = np.zeros_like(first_preds, dtype=np.float64)

    total_weight = 0.0

    for name, weight in weight_dict.items():
        if name not in preds_dict:
            raise KeyError(
                f"Stream '{name}' found in weights but missing from predictions dictionary."
            )

        preds = np.array(preds_dict[name])
        final_preds += weight * preds
        total_weight += weight

    # Sanity check for weights summing to 1 (approx)
    if not np.isclose(total_weight, 1.0, rtol=1e-5):
        print(f"Warning: Applied weights sum to {total_weight}, not 1.0.")

    return final_preds
