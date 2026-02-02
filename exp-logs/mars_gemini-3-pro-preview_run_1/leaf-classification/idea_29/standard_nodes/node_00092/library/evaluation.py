import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library import config


def compute_log_loss(y_true, y_pred, eps=1e-15):
    """
    Computes the multi-class log loss after clipping probabilities.

    This function applies the specific clipping rule: max(min(p, 1-10^-15), 10^-15)
    to avoid infinite loss values, mirroring the competition metric.

    Args:
        y_true (array-like): True class labels (encoded as integers).
        y_pred (array-like): Predicted probabilities of shape (n_samples, n_classes).
        eps (float): Epsilon value for clipping probabilities. Defaults to 1e-15.

    Returns:
        float: The computed log loss.
    """
    # Clip probabilities to avoid log(0) extremes
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

    # Compute log loss
    # sklearn.metrics.log_loss handles integer y_true and probability matrix y_pred
    # assuming y_true contains indices 0 to n_classes-1
    loss = log_loss(y_true, y_pred_clipped)

    # Print full precision as requested
    print(f"Validation Multi-class Log Loss: {loss}")

    return loss


def save_submission(ids, probs, class_names, output_path=None):
    """
    Formats and saves the submission file with clipped probabilities.

    Args:
        ids (array-like): The image IDs for the test set.
        probs (array-like): Predicted probabilities of shape (n_samples, n_classes).
        class_names (array-like): The list of class names corresponding to columns.
        output_path (str, optional): Path to save the CSV. If None, uses config.SUBMISSION_FILE.
    """
    if output_path is None:
        output_path = config.SUBMISSION_FILE

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Apply clipping to probabilities before saving
    # The task description notes that probabilities are replaced to avoid log extremes
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1 - eps)

    # Create DataFrame
    submission_df = pd.DataFrame(probs_clipped, columns=class_names)

    # Insert the ID column at the start
    # Ensure IDs are integers (e.g., 1202 instead of 1202.0)
    submission_df.insert(0, config.ID_COL, ids.astype(int))

    # Save to CSV
    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)
    print("Submission saved successfully.")
