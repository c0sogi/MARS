import os
import numpy as np
import pandas as pd


def weighted_soft_voting(prediction_list, weights):
    """
    Computes the weighted average of probability matrices from multiple models.

    Args:
        prediction_list (list of np.ndarray): List of probability matrices.
                                              Each matrix should have shape (n_samples, n_classes).
        weights (list of float): List of scalar weights corresponding to each model in prediction_list.

    Returns:
        np.ndarray: The weighted average probability matrix of shape (n_samples, n_classes).
    """
    # Input validation
    if len(prediction_list) != len(weights):
        raise ValueError(
            f"Mismatch: {len(prediction_list)} predictions vs {len(weights)} weights."
        )

    if not prediction_list:
        raise ValueError("Prediction list is empty.")

    # Ensure inputs are numpy arrays
    probs_arrays = [np.array(p) for p in prediction_list]

    # Check shapes consistency
    base_shape = probs_arrays[0].shape
    for i, p in enumerate(probs_arrays):
        if p.shape != base_shape:
            raise ValueError(
                f"Shape mismatch at index {i}: expected {base_shape}, got {p.shape}"
            )

    # Initialize weighted sum
    weighted_sum = np.zeros_like(probs_arrays[0], dtype=np.float64)
    total_weight = sum(weights)

    if total_weight <= 0:
        raise ValueError("Sum of weights must be positive.")

    # Accumulate weighted probabilities
    for p, w in zip(probs_arrays, weights):
        weighted_sum += p * w

    # Normalize
    ensemble_probs = weighted_sum / total_weight

    return ensemble_probs


def save_submission(test_ids, probabilities, class_labels, output_path):
    """
    Generates the submission CSV file by converting probabilities to class labels.

    Args:
        test_ids (pd.Series or np.ndarray): The IDs for the test samples.
        probabilities (np.ndarray): The final ensemble probabilities (n_samples, n_classes).
        class_labels (list or np.ndarray): The list of original class labels (e.g., [1, 2, 3...])
                                           corresponding to the column indices of the probability matrix.
                                           This is typically `le.classes_`.
        output_path (str): The file path where the submission CSV will be saved.
    """
    # Get the index of the class with the highest probability
    pred_indices = np.argmax(probabilities, axis=1)

    # Map indices back to original class labels
    # We use a numpy array for efficient indexing
    classes_arr = np.array(class_labels)
    final_predictions = classes_arr[pred_indices]

    # Create DataFrame
    # Using np.array(test_ids) ensures we don't carry over any index from a pandas Series
    submission_df = pd.DataFrame(
        {"Id": np.array(test_ids), "Cover_Type": final_predictions}
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
