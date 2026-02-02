import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def find_optimal_weights(predictions_list, true_labels):
    """
    Finds the optimal weights for a list of model predictions that minimize the Log Loss
    against the true labels using Constrained Convex Optimization (SLSQP).

    Args:
        predictions_list (list of np.ndarray): A list where each element is a 1D numpy array
                                               of probabilities from a specific model.
                                               All arrays must have the same length.
        true_labels (np.ndarray): A 1D numpy array of ground truth binary labels (0 or 1).

    Returns:
        np.ndarray: An array of scalar weights, one for each model, summing to 1.
    """
    num_models = len(predictions_list)

    # Validation
    if num_models == 0:
        raise ValueError("predictions_list cannot be empty.")

    # Ensure all predictions are numpy arrays
    predictions_list = [np.array(p) for p in predictions_list]

    # Objective function to minimize: Log Loss of the weighted average
    def loss_function(weights):
        # Calculate weighted average
        # We use a loop or broadcasting. Since num_models is small (15), a loop is clear.
        weighted_preds = np.zeros_like(predictions_list[0])
        for i in range(num_models):
            weighted_preds += weights[i] * predictions_list[i]

        # Clip predictions to avoid log(0) errors, though sklearn handles this internally usually.
        # Explicit clipping ensures stability within the optimizer.
        weighted_preds = np.clip(weighted_preds, 1e-15, 1 - 1e-15)

        return log_loss(true_labels, weighted_preds)

    # Constraint: Sum of weights must be 1
    # 'eq' means the function must equal 0
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Bounds: Weights must be between 0 and 1
    bounds = [(0.0, 1.0) for _ in range(num_models)]

    # Initial guess: Equal weights
    initial_weights = np.ones(num_models) / num_models

    # Run optimization
    result = minimize(
        loss_function,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": False, "maxiter": 1000},
    )

    if not result.success:
        print(f"Warning: Optimization did not converge. Message: {result.message}")

    optimal_weights = result.x

    # Normalize again just to be safe regarding floating point errors
    optimal_weights = optimal_weights / np.sum(optimal_weights)

    return optimal_weights


def weighted_average(predictions_list, weights):
    """
    Computes the weighted average of a list of prediction arrays.

    Args:
        predictions_list (list of np.ndarray): List of prediction arrays.
        weights (list or np.ndarray): List of scalar weights corresponding to predictions_list.

    Returns:
        np.ndarray: The weighted average prediction array.
    """
    if len(predictions_list) != len(weights):
        raise ValueError("Length of predictions_list and weights must match.")

    if len(predictions_list) == 0:
        return np.array([])

    # Initialize accumulator
    final_predictions = np.zeros_like(predictions_list[0])

    for pred, weight in zip(predictions_list, weights):
        final_predictions += weight * pred

    return final_predictions
