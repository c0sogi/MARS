import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss


def optimize_ensemble_weights(probs_a, probs_b, labels):
    """
    Finds the optimal scalar weight w for the ensemble: P = w * P_a + (1 - w) * P_b
    Minimizes Log Loss on the validation set using scalar minimization.

    Args:
        probs_a (np.ndarray): Probabilities from Stream A (N, C).
        probs_b (np.ndarray): Probabilities from Stream B (N, C).
        labels (np.ndarray): True label indices (N,).

    Returns:
        float: The optimal weight for Stream A (w).
    """
    print("Optimizing ensemble weights...")

    # Define the objective function to minimize (Log Loss)
    def objective(w):
        # Calculate weighted probabilities
        # w is the weight for Stream A, (1-w) for Stream B
        # Since both probs_a and probs_b sum to 1, the linear combination
        # w*A + (1-w)*B also sums to 1.
        probs_ensemble = w * probs_a + (1.0 - w) * probs_b

        # Calculate Log Loss
        # sklearn log_loss handles clipping internally to avoid log(0)
        return log_loss(labels, probs_ensemble)

    # Use bounded scalar minimization to find the best w in [0, 1]
    # This is more precise and often faster than a grid search
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")

    best_w = result.x
    best_loss = result.fun

    print(f"Optimization complete.")
    print(f"Optimal Ensemble Weight (Stream A): {best_w}")
    print(f"Best Combined Validation Log Loss: {best_loss}")

    return best_w


def apply_ensemble(probs_a, probs_b, weight_a):
    """
    Combines probability predictions from two streams using the specified weight.
    P_final = weight_a * P_a + (1 - weight_a) * P_b

    Args:
        probs_a (np.ndarray): Probabilities from Stream A.
        probs_b (np.ndarray): Probabilities from Stream B.
        weight_a (float): Weight for Stream A (0.0 to 1.0).

    Returns:
        np.ndarray: Combined probabilities.
    """
    # Calculate weighted average
    probs_final = weight_a * probs_a + (1.0 - weight_a) * probs_b

    return probs_final
