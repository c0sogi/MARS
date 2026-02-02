import numpy as np
from scipy.optimize import minimize_scalar
from library.utils import compute_log_loss
from library.config import Config


def blend_predictions(probs_linear, probs_transformer, weight_transformer):
    """
    Computes the weighted average of predictions from two models.

    Args:
        probs_linear (numpy.ndarray): Probabilities from the linear model.
        probs_transformer (numpy.ndarray): Probabilities from the transformer model.
        weight_transformer (float): Weight assigned to the transformer model (0 to 1).
                                    The linear model gets (1 - weight_transformer).

    Returns:
        numpy.ndarray: Blended probabilities.
    """
    weight_linear = 1.0 - weight_transformer
    return (probs_transformer * weight_transformer) + (probs_linear * weight_linear)


def optimize_weights(val_probs_linear, val_probs_transformer, y_val):
    """
    Finds the optimal weight for the transformer model that minimizes log loss
    on the validation set.

    Args:
        val_probs_linear (numpy.ndarray): Validation predictions from linear model.
        val_probs_transformer (numpy.ndarray): Validation predictions from transformer model.
        y_val (numpy.ndarray): Ground truth validation labels (integers).

    Returns:
        float: Optimal weight for the transformer model.
    """
    print("Optimizing ensemble weights...")

    # Define the objective function to minimize
    def objective(w):
        # Blend predictions with current weight w
        blended = blend_predictions(val_probs_linear, val_probs_transformer, w)
        # Compute log loss
        # We pass labels=[0, 1, 2] to ensure correct handling of classes
        return compute_log_loss(y_val, blended, labels=[0, 1, 2])

    # Minimize the objective function w.r.t w in the range [0, 1]
    # We use 'bounded' method because weights must be valid probabilities
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")

    best_weight = result.x
    best_loss = result.fun

    print(f"Optimization Complete.")
    print(f"Best Transformer Weight: {best_weight}")
    print(f"Best Linear Weight: {1.0 - best_weight}")
    print(f"Best Validation Log Loss: {best_loss}")

    return best_weight


def run_ensemble(
    val_probs_linear,
    test_probs_linear,
    val_probs_transformer,
    test_probs_transformer,
    y_val,
):
    """
    Orchestrates the ensemble process: optimizes weights on validation data
    and generates final blended predictions for the test set.

    Args:
        val_probs_linear (numpy.ndarray): Validation probs (Linear).
        test_probs_linear (numpy.ndarray): Test probs (Linear).
        val_probs_transformer (numpy.ndarray): Validation probs (Transformer).
        test_probs_transformer (numpy.ndarray): Test probs (Transformer).
        y_val (numpy.ndarray): Ground truth validation labels.

    Returns:
        numpy.ndarray: Final blended test probabilities.
    """
    print("--- Starting Ensemble Pipeline ---")

    # 1. Optimize Weights
    best_weight = optimize_weights(val_probs_linear, val_probs_transformer, y_val)

    # 2. Blend Test Predictions
    print("Blending test predictions with optimized weights...")
    final_test_probs = blend_predictions(
        test_probs_linear, test_probs_transformer, best_weight
    )

    # 3. Verify Validation Score (Sanity Check)
    # Re-calculate to confirm consistency
    final_val_probs = blend_predictions(
        val_probs_linear, val_probs_transformer, best_weight
    )
    val_loss = compute_log_loss(y_val, final_val_probs, labels=[0, 1, 2])
    print(f"Final Ensemble Validation Log Loss: {val_loss}")

    print("--- Ensemble Pipeline Complete ---")
    return final_test_probs
