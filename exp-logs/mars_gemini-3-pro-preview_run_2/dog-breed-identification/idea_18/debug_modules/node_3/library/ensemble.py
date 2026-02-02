import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
import library.config as config
import library.classifier as classifier
import library.data_utils as data_utils


def optimize_ensemble_weights(load_cached_data=True):
    """
    Calculates the optimal scalar weights for the ensemble of Stream A and Stream B
    by minimizing the Log Loss on the validation set.

    Args:
        load_cached_data (bool): Whether to load underlying predictions from cache.

    Returns:
        tuple: (weight_a, weight_b) corresponding to Stream A and Stream B.
    """
    print("Starting ensemble weight optimization...")

    # 1. Get Validation Predictions for Stream A (ConvNeXt)
    probs_a, ids_a, y_a = classifier.predict_stream(
        config.STREAM_A, split="val", load_cached_data=load_cached_data
    )

    # 2. Get Validation Predictions for Stream B (RegNet)
    probs_b, ids_b, y_b = classifier.predict_stream(
        config.STREAM_B, split="val", load_cached_data=load_cached_data
    )

    # 3. Verify Data Alignment
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("Validation IDs for Stream A and Stream B do not match.")

    if not np.array_equal(y_a, y_b):
        raise ValueError("Validation labels for Stream A and Stream B do not match.")

    # 4. Define Objective Function
    # We minimize Log Loss w.r.t weight w_a, where w_b = 1 - w_a
    y_true = y_a

    def objective(w_a):
        w_b = 1.0 - w_a
        # Calculate weighted probabilities
        p_ensemble = (w_a * probs_a) + (w_b * probs_b)

        # Numerical stability: clip probabilities to avoid log(0)
        # sklearn log_loss handles this, but explicit clipping ensures stability in optimization
        p_ensemble = np.clip(p_ensemble, 1e-15, 1 - 1e-15)

        # Renormalize rows to sum to 1 (clipping might slightly alter sums)
        p_ensemble /= p_ensemble.sum(axis=1, keepdims=True)

        # Cite debug_lesson_9: Explicitly pass labels to handle sparse subsets
        return log_loss(y_true, p_ensemble, labels=np.arange(probs_a.shape[1]))

    # 5. Optimize
    # Use bounded optimization to ensure weights are valid probabilities [0, 1]
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")

    best_w_a = result.x
    best_w_b = 1.0 - best_w_a
    best_loss = result.fun

    print("Optimization Complete.")
    print(f"Optimal Weight Stream A (ConvNeXt): {best_w_a}")
    print(f"Optimal Weight Stream B (RegNet): {best_w_b}")
    print(f"Best Validation Log Loss: {best_loss}")

    return best_w_a, best_w_b


def generate_submission(weights, load_cached_data=True):
    """
    Generates the submission file using the provided ensemble weights on the test set.

    Args:
        weights (tuple): (weight_a, weight_b)
        load_cached_data (bool): Whether to load underlying predictions from cache.
    """
    w_a, w_b = weights
    print(f"Generating submission with weights A={w_a}, B={w_b}...")

    # 1. Get Test Predictions for Stream A
    probs_a, ids_a, _ = classifier.predict_stream(
        config.STREAM_A, split="test", load_cached_data=load_cached_data
    )

    # 2. Get Test Predictions for Stream B
    probs_b, ids_b, _ = classifier.predict_stream(
        config.STREAM_B, split="test", load_cached_data=load_cached_data
    )

    # 3. Verify Alignment
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("Test IDs for Stream A and Stream B do not match.")

    # 4. Compute Weighted Average
    final_probs = (w_a * probs_a) + (w_b * probs_b)

    # 5. Prepare Submission DataFrame
    # Get breed names in correct order (alphabetical, as per data_utils)
    _, breed_names = data_utils.get_label_map()

    # Create DataFrame with breed columns
    df = pd.DataFrame(final_probs, columns=breed_names)

    # Insert ID column at the start
    df.insert(0, "id", ids_a)

    # 6. Save to CSV
    save_path = config.SUBMISSION_PATH
    print(f"Saving submission to {save_path}...")
    df.to_csv(save_path, index=False)
    print("Submission saved successfully.")
