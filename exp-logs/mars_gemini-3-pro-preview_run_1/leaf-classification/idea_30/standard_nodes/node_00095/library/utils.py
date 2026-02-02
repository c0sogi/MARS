import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from library.config import SUBMISSION_FILE_PATH, PROB_CLIP_MIN, PROB_CLIP_MAX


def save_submission(predictions, test_ids, class_names):
    """
    Formats and saves the submission file.

    Args:
        predictions (np.ndarray): Array of shape (n_samples, n_classes) containing probabilities.
        test_ids (np.ndarray): Array of shape (n_samples,) containing image IDs.
        class_names (list or np.ndarray): List of class names corresponding to the columns of predictions.
    """
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    test_ids = np.array(test_ids).flatten()

    # Create a dictionary for the DataFrame
    data = {"id": test_ids}

    # Add probability columns
    for i, class_name in enumerate(class_names):
        data[class_name] = predictions[:, i]

    # Create DataFrame
    submission_df = pd.DataFrame(data)

    # Save to CSV
    submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_FILE_PATH}")


def compute_log_loss(y_true, y_pred, class_names):
    """
    Computes the multi-class log loss metric with specific rescaling and clipping.

    Args:
        y_true (np.ndarray or list): Ground truth labels (strings or integers).
        y_pred (np.ndarray): Predicted probabilities of shape (n_samples, n_classes).
        class_names (list or np.ndarray): List of all unique class names ordered by column index of y_pred.

    Returns:
        float: The calculated log loss.
    """
    # Ensure y_pred is a numpy array
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale: Skipped to avoid floating point noise (Cite solution_lesson_node_00085).
    # The model output (Softmax) is already normalized.

    # 2. Clip: Replace probabilities with max(min(p, 1-10^-15), 10^-15)
    y_pred_clipped = np.clip(y_pred, PROB_CLIP_MIN, PROB_CLIP_MAX)

    # 3. Compute Log Loss
    # sklearn log_loss handles string labels in y_true if 'labels' is provided
    loss = log_loss(y_true, y_pred_clipped, labels=class_names)

    return loss
