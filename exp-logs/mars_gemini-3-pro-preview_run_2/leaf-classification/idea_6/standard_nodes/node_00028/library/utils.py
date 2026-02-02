import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def clip_probabilities(probs):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] to ensure numerical stability
    and adhere to the competition metric definition.

    Args:
        probs (np.ndarray or pd.DataFrame): The probability matrix.

    Returns:
        np.ndarray: The clipped probability matrix.
    """
    epsilon = 1e-15
    # Convert to numpy array if it's not already
    probs_arr = np.array(probs)
    return np.clip(probs_arr, epsilon, 1 - epsilon)


def calculate_log_loss(y_true, y_pred, class_labels=None):
    """
    Calculates the multi-class log loss metric according to the competition specifications.
    This includes row-wise normalization and clipping prior to scoring.

    Args:
        y_true (array-like): True class labels (strings or integers).
        y_pred (array-like): Predicted probabilities (n_samples, n_classes).
        class_labels (list, optional): List of class labels corresponding to the columns of y_pred.
                                       Required if y_true contains string labels to ensure correct mapping.

    Returns:
        float: The calculated log loss.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred)

    # 1. Rescale: Each row is divided by the row sum
    row_sums = np.sum(y_pred, axis=1)
    # Handle potential division by zero (though unlikely with proper models)
    row_sums[row_sums == 0] = 1.0
    y_pred_norm = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: Replace probabilities with max(min(p, 1-10^-15), 10^-15)
    y_pred_clipped = clip_probabilities(y_pred_norm)

    # 3. Calculate Log Loss
    # sklearn's log_loss handles the label encoding if labels are provided
    return log_loss(y_true, y_pred_clipped, labels=class_labels)


def save_submission(ids, class_names, probs, output_path):
    """
    Formats predictions into the required CSV format and saves the file.

    Args:
        ids (array-like): Sequence of image IDs.
        class_names (list): Sequence of species names corresponding to the probability columns.
        probs (np.ndarray): Matrix of predicted probabilities (n_samples, n_classes).
        output_path (str): File path where the submission CSV will be saved.
    """
    # Validate dimensions
    if len(ids) != len(probs):
        raise ValueError(
            f"Length of IDs ({len(ids)}) does not match number of predictions ({len(probs)})."
        )
    if len(class_names) != probs.shape[1]:
        raise ValueError(
            f"Number of class names ({len(class_names)}) does not match probability columns ({probs.shape[1]})."
        )

    # Create DataFrame
    submission_df = pd.DataFrame(probs, columns=class_names)

    # Insert 'id' as the first column
    submission_df.insert(0, "id", ids)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
