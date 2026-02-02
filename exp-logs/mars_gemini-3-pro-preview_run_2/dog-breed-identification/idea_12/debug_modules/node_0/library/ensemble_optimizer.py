import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from library.config import Config


def optimize_ensemble_weights(preds_a, preds_b, true_labels, class_names):
    """
    Finds the optimal scalar weight for blending two sets of predictions
    to minimize Log Loss on the validation set.

    Args:
        preds_a (np.ndarray): Probabilities from Stream A (N_samples, N_classes).
        preds_b (np.ndarray): Probabilities from Stream B (N_samples, N_classes).
        true_labels (np.ndarray): Ground truth labels (N_samples,). Can be strings or indices.
        class_names (list or np.ndarray): List of class names corresponding to the columns
                                          of preds_a and preds_b.

    Returns:
        float: Optimal weight for Stream A (weight for Stream B is 1 - weight_a).
    """
    print("Optimizing ensemble weights...")

    # Ensure inputs are numpy arrays
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    true_labels = np.array(true_labels)

    # Clip predictions to avoid log(0) errors, though log_loss handles this internally usually.
    # We rely on sklearn's internal handling but ensuring validity is good practice.
    eps = 1e-15
    preds_a = np.clip(preds_a, eps, 1 - eps)
    preds_b = np.clip(preds_b, eps, 1 - eps)

    def objective(w_a):
        """
        Objective function to minimize.
        w_a: Weight for Stream A.
        w_b: 1 - w_a
        """
        # Blend predictions
        w_b = 1.0 - w_a
        preds_blended = (w_a * preds_a) + (w_b * preds_b)

        # Normalize to ensure sum to 1 (handling potential float precision issues)
        # axis=1 sum might drift slightly from 1.0
        preds_blended = preds_blended / preds_blended.sum(axis=1, keepdims=True)

        # Calculate Log Loss
        # labels parameter is crucial if true_labels are strings or if not all classes are present in val
        loss = log_loss(true_labels, preds_blended, labels=class_names)
        return loss

    # Use bounded optimization to find w_a in [0, 1]
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")

    best_w_a = result.x
    best_loss = result.fun

    print(f"Optimization complete.")
    print(f"Best Weight for Stream A: {best_w_a}")
    print(f"Best Weight for Stream B: {1.0 - best_w_a}")
    print(f"Combined Validation Log Loss: {best_loss}")

    return best_w_a


def blend_predictions(preds_a, preds_b, weight_a):
    """
    Blends two prediction arrays using the specified weight.

    Args:
        preds_a (np.ndarray): Predictions from Stream A.
        preds_b (np.ndarray): Predictions from Stream B.
        weight_a (float): Weight for Stream A.

    Returns:
        np.ndarray: Blended predictions.
    """
    weight_b = 1.0 - weight_a

    # Weighted sum
    blended = (weight_a * preds_a) + (weight_b * preds_b)

    # Renormalize to ensure valid probability distribution
    # (Fixes minor floating point errors)
    row_sums = blended.sum(axis=1, keepdims=True)
    blended = blended / row_sums

    return blended


def generate_submission(
    test_ids, predictions, class_names, output_path=Config.SUBMISSION_FILE
):
    """
    Generates the submission CSV file.

    Args:
        test_ids (np.ndarray): Array of test image IDs.
        predictions (np.ndarray): Matrix of predicted probabilities (N_test, N_classes).
        class_names (list): List of class names corresponding to columns.
        output_path (str): Path to save the submission file.
    """
    print(f"Generating submission file at {output_path}...")

    # Create DataFrame
    df = pd.DataFrame(predictions, columns=class_names)

    # Insert ID column at the beginning
    df.insert(0, "id", test_ids)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"Submission saved successfully. Shape: {df.shape}")
