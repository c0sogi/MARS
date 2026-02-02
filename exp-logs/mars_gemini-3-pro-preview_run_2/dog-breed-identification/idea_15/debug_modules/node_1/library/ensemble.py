import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from library.config import SUBMISSION_PATH, NUM_CLASSES, STREAMS
from library.dataset import get_class_mapping
from library.training import train_stream


def optimize_ensemble_weights(val_preds_a, val_preds_b, val_labels):
    """
    Finds the optimal scalar weights for blending two streams of probability predictions
    to minimize Log Loss on the validation set.

    Args:
        val_preds_a (np.ndarray): Validation probabilities from Stream A (N, C).
        val_preds_b (np.ndarray): Validation probabilities from Stream B (N, C).
        val_labels (np.ndarray): Ground truth labels (N,).

    Returns:
        tuple: (weight_a, weight_b) where weight_a + weight_b = 1.
    """
    print("Optimizing ensemble weights...")

    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    p_a = np.clip(val_preds_a, epsilon, 1 - epsilon)
    p_b = np.clip(val_preds_b, epsilon, 1 - epsilon)

    def objective(w):
        # Linear combination: P = w * P_a + (1-w) * P_b
        p_blend = w * p_a + (1 - w) * p_b
        # Re-clip after blending to be safe
        p_blend = np.clip(p_blend, epsilon, 1 - epsilon)
        return log_loss(val_labels, p_blend)

    # Optimize w in [0, 1]
    # We use minimize_scalar with 'bounded' method
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")

    best_w_a = result.x
    best_w_b = 1.0 - best_w_a
    best_loss = result.fun

    print(f"Optimization complete.")
    print(f"Optimal Weight A (ConvNeXt): {best_w_a:.10f}")
    print(f"Optimal Weight B (MaxViT):   {best_w_b:.10f}")
    print(f"Best Validation Log Loss:    {best_loss:.10f}")

    return best_w_a, best_w_b


def generate_submission(test_preds_a, test_preds_b, w_a, w_b, test_ids):
    """
    Generates the final submission file by blending test predictions.

    Args:
        test_preds_a (np.ndarray): Test probabilities from Stream A.
        test_preds_b (np.ndarray): Test probabilities from Stream B.
        w_a (float): Weight for Stream A.
        w_b (float): Weight for Stream B.
        test_ids (np.ndarray): Array of test image IDs.
    """
    print("Generating submission file...")

    # Weighted average
    final_probs = w_a * test_preds_a + w_b * test_preds_b

    # Get column names (breeds)
    # The dataset module provides a mapping. We need index -> breed.
    _, idx_to_breed = get_class_mapping()
    breed_cols = [idx_to_breed[i] for i in range(NUM_CLASSES)]

    # Create DataFrame
    df = pd.DataFrame(final_probs, columns=breed_cols)
    df.insert(0, "id", test_ids)

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def run_ensemble(load_cached_model=True):
    """
    Orchestrates the ensemble pipeline:
    1. Gets predictions from both streams (training/loading models as needed).
    2. Optimizes weights on validation set.
    3. Generates submission on test set.

    Args:
        load_cached_model (bool): Passed to model training to decide whether to load saved heads.
    """
    # 1. Get predictions from Stream A (ConvNeXt)
    print("\n=== Stream A: ConvNeXt ===")
    res_a = train_stream(STREAMS["stream_a"], load_cached_model=load_cached_model)

    # 2. Get predictions from Stream B (MaxViT)
    print("\n=== Stream B: MaxViT ===")
    res_b = train_stream(STREAMS["stream_b"], load_cached_model=load_cached_model)

    # Validate alignment
    if not np.array_equal(res_a["val_y"], res_b["val_y"]):
        raise ValueError("Validation labels mismatch between streams.")
    if not np.array_equal(res_a["test_ids"], res_b["test_ids"]):
        raise ValueError("Test IDs mismatch between streams.")

    # 3. Optimize Weights
    w_a, w_b = optimize_ensemble_weights(
        res_a["val_probs"], res_b["val_probs"], res_a["val_y"]
    )

    # 4. Generate Submission
    generate_submission(
        res_a["test_probs"], res_b["test_probs"], w_a, w_b, res_a["test_ids"]
    )
