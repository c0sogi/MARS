import numpy as np
from library.utils import calculate_log_loss, save_submission
from library.config import Config


def optimize_ensemble_weights(p_val_stat, p_val_dl, y_val):
    """
    Finds the optimal weight for blending Statistical and Transformer predictions
    to minimize Log Loss on the validation set.

    Formula: P_final = w * P_dl + (1 - w) * P_stat

    Args:
        p_val_stat (np.ndarray): Validation probabilities from Statistical model (n_samples, n_classes).
        p_val_dl (np.ndarray): Validation probabilities from Transformer model (n_samples, n_classes).
        y_val (np.ndarray): True validation labels (indices).

    Returns:
        float: Optimal weight for the Transformer component.
    """
    best_loss = float("inf")
    best_w = 0.5

    # Search space: 0.0 to 1.0 with step 0.01
    # We test 101 points to cover 0.00, 0.01, ..., 1.00
    search_space = np.linspace(0, 1, 101)

    print("Optimizing ensemble weights...")

    for w in search_space:
        # Blend predictions
        # w is the weight for the Deep Learning (Transformer) model
        p_blend = w * p_val_dl + (1 - w) * p_val_stat

        # Calculate loss
        loss = calculate_log_loss(y_val, p_blend)

        if loss < best_loss:
            best_loss = loss
            best_w = w

    print(f"Optimal Transformer Weight: {best_w}")
    print(f"Best Ensemble Validation Log Loss: {best_loss}")

    return best_w


def apply_ensemble(p_stat, p_dl, weight):
    """
    Applies the weighted ensemble to predictions.

    Args:
        p_stat (np.ndarray): Probabilities from Statistical model.
        p_dl (np.ndarray): Probabilities from Transformer model.
        weight (float): Weight for the Transformer model.

    Returns:
        np.ndarray: Blended probabilities.
    """
    return weight * p_dl + (1 - weight) * p_stat


def generate_and_save_submission(
    test_ids, p_test_stat, p_test_dl, weight, output_path=Config.SUBMISSION_PATH
):
    """
    Generates the final ensemble predictions and saves them to a CSV file.

    Args:
        test_ids (list or np.ndarray): IDs for the test samples.
        p_test_stat (np.ndarray): Test probabilities from Statistical model.
        p_test_dl (np.ndarray): Test probabilities from Transformer model.
        weight (float): Optimal weight for the Transformer model.
        output_path (str): Path to save the submission CSV.
    """
    print(f"Generating ensemble predictions with Transformer weight: {weight}")

    # Calculate blended probabilities
    final_probs = apply_ensemble(p_test_stat, p_test_dl, weight)

    print(f"Saving submission to {output_path}...")

    # Use the utility function from library.utils to handle formatting and clipping
    save_submission(test_ids, final_probs, output_path)

    print("Submission saved successfully.")
