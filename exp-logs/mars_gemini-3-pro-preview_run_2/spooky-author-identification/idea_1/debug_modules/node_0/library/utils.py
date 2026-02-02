import os
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from library.config import Config


def compute_log_loss(y_true, y_pred):
    """
    Computes the multi-class logarithmic loss according to the competition metric.

    Args:
        y_true (array-like): Ground truth labels. Can be strings (e.g., 'EAP') corresponding
                             to Config.CLASSES, or integers if aligned.
        y_pred (array-like): Predicted probabilities with shape (n_samples, n_classes).
                             The columns must correspond to the order in Config.CLASSES.

    Returns:
        float: The calculated log loss.
    """
    # Calculate log loss.
    # eps=1e-15 handles the clipping requirement: max(min(p, 1-10^-15), 10^-15).
    # labels=Config.CLASSES ensures correct mapping between y_pred columns and string labels in y_true.
    loss = log_loss(y_true, y_pred, eps=1e-15, labels=Config.CLASSES)
    return loss


def save_submission(ids, predictions, filename=None):
    """
    Formats and saves the predictions to a CSV file in the required submission format.

    Args:
        ids (array-like): Sequence of sample IDs (e.g., ['id001', 'id002']).
        predictions (array-like): Predicted probabilities matrix of shape (n_samples, 3).
                                  Column order must match Config.CLASSES ['EAP', 'HPL', 'MWS'].
        filename (str, optional): The file path to save the submission.
                                  Defaults to Config.SUBMISSION_FILE.
    """
    if filename is None:
        filename = Config.SUBMISSION_FILE

    # Ensure the output directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Create DataFrame with correct column headers
    submission_df = pd.DataFrame(predictions, columns=Config.CLASSES)

    # Insert the 'id' column at the beginning
    submission_df.insert(0, Config.ID_COL, ids)

    # Save to CSV without the pandas index
    submission_df.to_csv(filename, index=False)

    print(f"Submission saved to {filename}")
